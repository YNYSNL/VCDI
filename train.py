import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from torch.utils.data import DataLoader
from datetime import datetime


class Inter_DrivingData(Dataset):
    def __init__(self, file_list):
        self.data_list = file_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Define the number of retry attempts
        retries = 3
        for attempt in range(retries):
            try:
                data = np.load(self.data_list[idx])
                ego = data['ego']
                neighbors = data['neighbors']
                ref_line = data['ref_line']
                map_lanes = data['map_lanes']
                map_crosswalks = data['map_crosswalks']
                gt_future_states = data['gt_future_states']

                return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states

            except Exception as e:
                print(
                    f"Attempt {attempt+1} - Error loading {self.data_list[idx]}: {e}")
                if attempt == retries - 1:
                    return None


def train_epoch(data_loader, predictor, planner, optimizer, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device)
        future_action = ground_truth[:, 0, :, :2]
        current_state = torch.cat(
            [ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        optimizer.zero_grad()
        plans, predictions, scores, cost_function_weights = predictor(
            ego, neighbors, map_lanes, map_crosswalks, future_action)
        plan_trajs = torch.stack(
            [bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
        # multi-future multi-agent loss
        loss = MFMA_loss(plan_trajs, predictions,
                         scores, ground_truth, weights)

        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                # initial control sequence
                "control_variables": plan.view(-1, 100),
                "predictions": prediction,  # prediction for surrounding vehicles
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(
                    1).to(args.device)

            final_values, info = planner.layer.forward(planner_inputs)
            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_metric().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1],
                                          ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost  # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 5)
        optimizer.step()

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show loss
        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rTrain Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(
        epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(
        epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(
        f'\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}')

    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device)
        future_action = ground_truth[:, 0, :, :2]
        current_state = torch.cat(
            [ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(
                ego, neighbors, map_lanes, map_crosswalks, future_action)
            plan_trajs = torch.stack(
                [bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            # multi-future multi-agent loss
            loss = MFMA_loss(plan_trajs, predictions,
                             scores, ground_truth, weights)

        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                # generate initial control sequence
                "control_variables": plan.view(-1, 100),
                "predictions": prediction,  # generate predictions for surrounding vehicles
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(
                    1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_metric().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1],
                                          ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost  # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(
            f"\rValid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(
        epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(
        epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(
        f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')

    return np.mean(epoch_loss), epoch_metrics


def model_training():
    # Logging
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.name = args.model_type + args.name + f"_{current_datetime}"
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info(
        "Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up predictor
    predictor = Predictor(50, model_type=args.model_type).to(args.device)

    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.device)
    else:
        planner = None

    # set up optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    set_dir = args.train_set
    files_pattern = os.path.join(set_dir, '*')
    all_files = glob.glob(files_pattern)

    # splitting data
    selected_index = int(len(all_files))
    selected_all_files = all_files[:selected_index]
    split_index_train = int(0.8 * len(selected_all_files))
    split_index_val = int(0.9 * len(selected_all_files))
    train_files_list = selected_all_files[:split_index_train]
    val_files_list = selected_all_files[split_index_train:split_index_val]
    test_files_list = selected_all_files[split_index_val:]
    save_path = log_path
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    file_save_path_all = os.path.join(save_path, 'selected_files_all.txt')
    file_save_path_train = os.path.join(save_path, 'selected_files_train.txt')
    file_save_path_val = os.path.join(save_path, 'selected_files_val.txt')
    file_save_path_test = os.path.join(save_path, 'selected_files_test.txt')
    # Save the selected_all_files list to a file
    with open(file_save_path_all, 'w') as file:
        for filepath in selected_all_files:
            file.write(filepath + '\n')

    with open(file_save_path_train, 'w') as file:
        for filepath in train_files_list:
            file.write(filepath + '\n')

    with open(file_save_path_val, 'w') as file:
        for filepath in val_files_list:
            file.write(filepath + '\n')

    with open(file_save_path_test, 'w') as file:
        for filepath in test_files_list:
            file.write(filepath + '\n')

    train_set = Inter_DrivingData(train_files_list)
    valid_set = Inter_DrivingData(val_files_list)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=32,
                              shuffle=False, num_workers=args.num_workers, drop_last=True)
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(
        len(train_set), len(valid_set)))

    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")

        # train
        if planner:
            if epoch < args.pretrain_epochs:
                args.use_planning = False
            else:
                args.use_planning = True

        train_loss, train_metrics = train_epoch(
            train_loader, predictor, planner, optimizer, args.use_planning)
        val_loss, val_metrics = valid_epoch(
            valid_loader, predictor, planner, args.use_planning)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss,
               'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1],
               'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
               'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1],
               'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(predictor.state_dict(
        ), f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str,
                        help='log name', default="Train1")
    parser.add_argument('--train_set', type=str, help='path to train datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers used for dataloader")
    parser.add_argument('--pretrain_epochs', type=int,
                        help='epochs of pretraining predictor', default=5)
    parser.add_argument('--train_epochs', type=int,
                        help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int,
                        help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--use_planning', action="store_true",
                        help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--model_type', type=str,
                        help='DIPP or VCDI or Gaussian', default='VCDI')
    parser.add_argument(
        '--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    model_training()
