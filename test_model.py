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
from utils.test_utils import *
from datetime import datetime


class Inter_DrivingData(Dataset):
    def __init__(self, file_list):

        self.data_list = file_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
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


def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def test_epoch(data_loader, predictor, planner, use_planning, device):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        map_lanes = batch[2].to(device)
        map_crosswalks = batch[3].to(device)
        ref_line_info = batch[4].to(device)
        ground_truth = batch[5].to(device)
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
                planner_inputs[f'cost_function_weight_{i + 1}'] = cost_function_weights[:, i].unsqueeze(
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
            f"\rTest Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time() - start_time) / current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(
        epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])

    predictorADE, predictorFDE_5, predictorFDE_3, predictorFDE_1 = np.mean(epoch_metrics[:, 2]), np.mean(
        epoch_metrics[:, 3]), np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE,
                     predictorFDE_5, predictorFDE_3, predictorFDE_1]
    logging.info(
        f'\ntest-plannerADE: {plannerADE:.4f}, test-plannerFDE: {plannerFDE:.4f}, test-predictorADE: {predictorADE:.4f}, test-predictorFDE_5: {predictorFDE_5:.4f}, test-predictorFDE_3: {predictorFDE_3:.4f}, test-predictorFDE_1: {predictorFDE_1:.4f}')


def model_testing():
    # Logging
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.name = args.model_type + args.name + f"_{current_datetime}"
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info(
        "Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    predictor = Predictor(50, model_type=args.model_type).to(args.device)
    predictor.load_state_dict(torch.load(
        args.model_path, map_location=args.device))
    trajectory_len, feature_len = 50, 9
    planner = MotionPlanner(trajectory_len, feature_len, args.device)

    test_files_list = read_file_to_list(args.test_set)
    test_set = Inter_DrivingData(test_files_list)

    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, num_workers=1)

    test_epoch(test_loader, predictor, planner,
               use_planning=True, device=args.device)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--name', type=str,
                        help='log name', default="Test1")
    parser.add_argument('--model_path', type=str,
                        help='path to the trained model')
    parser.add_argument('--test_set', type=str,
                        help='path to the list of test datasets')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers used for dataloader")
    parser.add_argument('--batch_size', type=int,
                        help='batch size (default: 32)', default=32)
    parser.add_argument('--use_planning', action="store_true",
                        help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--model_type', type=str,
                        help='DIPP or VCDI or Gaussian', default='VCDI')
    parser.add_argument(
        '--device', type=str, help='run on which device (default: cuda)', default='cuda:0')
    args = parser.parse_args()

    # Run
    model_testing()
