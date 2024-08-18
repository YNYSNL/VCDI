import numpy as np
import torch
import argparse
import glob
import os
import logging
import time
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from utils.test_utils import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from waymo_open_dataset.protos import scenario_pb2
from matplotlib.lines import Line2D
from datetime import datetime


def open_loop_test():
    # logging
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
    args.name = args.model_type + args.name + f"_{current_datetime}"
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info(
        "Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # test file
    files = glob.glob(args.open_loop_test_set+'/*')
    processor = TestDataProcess()

    # cache results
    collisions = []
    red_light, off_route = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE_1, prediction_FDE_3, prediction_FDE_5 = [], [], [], []

    # load model
    predictor = Predictor(50, model_type=args.model_type).to(args.device)
    predictor.load_state_dict(torch.load(
        args.model_path, map_location=args.device))
    predictor.eval()

    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(
            trajectory_len, feature_len, device=args.device, test=True)

    # iterate test files
    for file in files:
        scenarios = tf.data.TFRecordDataset(file)

        # iterate scenarios in the test file
        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())

            scenario_id = parsed_data.scenario_id

            sdc_id = parsed_data.sdc_track_index
            timesteps = parsed_data.timestamps_seconds

            # build map
            processor.build_map(parsed_data.map_features,
                                parsed_data.dynamic_map_states)
            logging.info(f"Scenario: {scenario_id}")
            # get a testing scenario
            for timestep in range(20, len(timesteps)-50, 10):
                # prepare data
                input_data = processor.process_frame(
                    timestep, sdc_id, parsed_data.tracks)
                ego = torch.from_numpy(input_data[0]).to(args.device)
                neighbors = torch.from_numpy(input_data[1]).to(args.device)
                lanes = torch.from_numpy(input_data[2]).to(args.device)
                crosswalks = torch.from_numpy(input_data[3]).to(args.device)
                ref_line = torch.from_numpy(input_data[4]).to(args.device)
                neighbor_ids, norm_gt_data, gt_data = input_data[5], input_data[6], input_data[7]
                future_action = torch.from_numpy(
                    input_data[6][0, :, :2]).unsqueeze(0).to(args.device)
                current_state = torch.cat(
                    [ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

                # predict
                with torch.no_grad():
                    plans, predictions, scores, cost_function_weights = predictor(
                        ego, neighbors, lanes, crosswalks, future_action)
                    plan, prediction = select_future(
                        plans, predictions, scores)

                weights_control = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
                cost_function_weights = cost_function_weights * weights_control

                # plan
                if args.use_planning:
                    planner_inputs = {
                        "control_variables": plan.view(-1, 100),
                        "predictions": prediction,
                        "ref_line_info": ref_line,
                        "current_state": current_state
                    }

                    for i in range(feature_len):
                        planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(
                            0)

                    with torch.no_grad():
                        final_values, info = planner.layer.forward(
                            planner_inputs, optimizer_kwargs={'track_best_solution': True})
                        plan = info.best_solution['control_variables'].view(
                            -1, 50, 2).to(args.device)

                plan = bicycle_model(plan, ego[:, -1])[:, :, :3]
                plan_control = plan[:, :, :2]

                # predict
                with torch.no_grad():
                    plans, predictions, scores, cost_function_weights = predictor(
                        ego, neighbors, lanes, crosswalks, plan_control)
                    plan, prediction = select_future(
                        plans, predictions, scores)

                weights_control = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
                cost_function_weights = cost_function_weights * weights_control
                # plan
                if args.use_planning:
                    planner_inputs = {
                        "control_variables": plan.view(-1, 100),
                        "predictions": prediction,
                        "ref_line_info": ref_line,
                        "current_state": current_state
                    }

                    for i in range(feature_len):
                        planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(
                            0)

                    with torch.no_grad():
                        final_values, info = planner.layer.forward(
                            planner_inputs, optimizer_kwargs={'track_best_solution': True})
                        plan = info.best_solution['control_variables'].view(
                            -1, 50, 2).to(args.device)

                plan = bicycle_model(plan, ego[:, -1])[:, :, :3]
                plan = plan.cpu().numpy()[0]

                # compute metrics
                logging.info(f"Results:")
                collision = check_collision(
                    plan, norm_gt_data[1:], current_state.cpu().numpy()[0, :, 5:])
                collisions.append(collision)
                traffic = check_traffic(plan, ref_line.cpu().numpy()[0])
                red_light.append(traffic[0])
                off_route.append(traffic[1])
                logging.info(
                    f"Collision: {collision}, Red light: {traffic[0]}, Off route: {traffic[1]}")

                Acc, Jerk, Lat_Acc = check_dynamics(plan)
                Accs.append(Acc)
                Jerks.append(Jerk)
                Lat_Accs.append(Lat_Acc)
                logging.info(
                    f"Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                Acc, Jerk, Lat_Acc = check_dynamics(norm_gt_data[0])
                Human_Accs.append(Acc)
                Human_Jerks.append(Jerk)
                Human_Lat_Accs.append(Lat_Acc)
                logging.info(
                    f"Human: Acceleration: {Acc}, Jerk: {Jerk}, Lateral_Acceleration: {Lat_Acc}")

                similarity = check_similarity(plan, norm_gt_data[0])
                similarity_1s.append(similarity[9])
                similarity_3s.append(similarity[29])
                similarity_5s.append(similarity[49])
                logging.info(
                    f"Similarity@1s: {similarity[9]}, Similarity@3s: {similarity[29]}, Similarity@5s: {similarity[49]}")

                prediction_error = check_prediction(
                    prediction[0].cpu().numpy(), norm_gt_data[1:])
                prediction_ADE.append(prediction_error[0])
                prediction_FDE_1.append(prediction_error[1])
                prediction_FDE_3.append(prediction_error[2])
                prediction_FDE_5.append(prediction_error[3])
                logging.info(
                    f"Prediction ADE: {prediction_error[0]}, FDE_1: {prediction_error[1]}, FDE_3: {prediction_error[2]}, FDE_5: {prediction_error[3]}")

                ### plot scenario ###
                if args.render:
                    # visualization
                    plt.ion()

                    # map
                    for vector in parsed_data.map_features:
                        vector_type = vector.WhichOneof("feature_data")
                        vector = getattr(vector, vector_type)
                        polyline = map_process(vector, vector_type)

                    # sdc
                    # [sdc, vehicle, pedestrian, cyclist]
                    agent_color = ['r', 'm', 'b', 'g']
                    color = agent_color[0]
                    track = parsed_data.tracks[sdc_id].states[timestep]
                    curr_state = (track.center_x,
                                  track.center_y, track.heading)
                    plan = transform(plan, curr_state, include_curr=True)

                    rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2),
                                         track.length, track.width, linewidth=2, color=color, alpha=0.6, zorder=3,
                                         transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    plt.plot(plan[::5, 0], plan[::5, 1], linewidth=1,
                             color=color, marker='*', markersize=3, zorder=3)
                    ego_gt = np.insert(
                        gt_data[0, :, :3], 0, curr_state, axis=0)
                    plt.plot(ego_gt[:, 0], ego_gt[:, 1],
                             'k--', linewidth=0.6, zorder=3)

                    # neighbors
                    for i, id in enumerate(neighbor_ids):
                        track = parsed_data.tracks[id].states[timestep]
                        color = agent_color[parsed_data.tracks[id].object_type]
                        rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2),
                                             track.length, track.width, linewidth=2, color=color, alpha=0.6, zorder=3,
                                             transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + plt.gca().transData)
                        plt.gca().add_patch(rect)

                        predict_traj = prediction.cpu().numpy()[0, i]
                        predict_traj = transform(predict_traj, curr_state)
                        predict_traj = np.insert(
                            predict_traj, 0, (track.center_x, track.center_y), axis=0)
                        plt.plot(predict_traj[::5, 0], predict_traj[::5, 1],
                                 linewidth=1, color='m', marker='.', markersize=3, zorder=3)

                        other_gt = np.insert(
                            gt_data[i+1, :, :3], 0, (track.center_x, track.center_y, track.heading), axis=0)
                        other_gt = other_gt[other_gt[:, 0] != 0]
                        plt.plot(other_gt[:, 0], other_gt[:, 1],
                                 'k--', linewidth=0.6, zorder=3)

                    for i, track in enumerate(parsed_data.tracks):
                        if i not in [sdc_id] + neighbor_ids and track.states[timestep].valid:
                            rect = plt.Rectangle((track.states[timestep].center_x-track.states[timestep].length/2, track.states[timestep].center_y-track.states[timestep].width/2),
                                                 track.states[timestep].length, track.states[timestep].width, linewidth=2, color='m', alpha=0.6, zorder=3,
                                                 transform=mpl.transforms.Affine2D().rotate_around(*(track.states[timestep].center_x, track.states[timestep].center_y), track.states[timestep].heading) + plt.gca().transData)
                            plt.gca().add_patch(rect)

                    # dynamic_map_states
                    for signal in parsed_data.dynamic_map_states[timestep].lane_states:
                        traffic_signal_process(processor.lanes, signal)

                    legend_proxy1 = Line2D(
                        [0], [0], linestyle="none", c=agent_color[0], marker='s')
                    legend_proxy2 = Line2D(
                        [0], [0], linestyle="none", c=agent_color[1], marker='s')
                    legend_proxy4 = Line2D(
                        [0], [0], linestyle="none", c=agent_color[2], marker='s')
                    legend_proxy5 = Line2D(
                        [0], [0], linestyle="none", c=agent_color[3], marker='s')

                    legend_proxy6 = Line2D(
                        [0], [0], linestyle='-', color=agent_color[0], marker='.')
                    legend_proxy7 = Line2D(
                        [0], [0], linestyle='-', color=agent_color[1], marker='.')
                    legend_proxy8 = Line2D(
                        [0], [0], linestyle='--', color='black', marker=None)
                    plt.legend([legend_proxy1, legend_proxy2, legend_proxy4, legend_proxy5, legend_proxy6, legend_proxy7, legend_proxy8], [
                               'AV', 'BV', 'Pedestrian', 'Cyclist', 'Planned', 'Predicted', 'Ground-truth'], numpoints=1)

                    # show plot
                    plt.gca().axis(
                        [-40 + plan[0, 0], 80 + plan[0, 0], -60 + plan[0, 1], 60 + plan[0, 1]])
                    plt.gca().set_facecolor('white')
                    plt.gca().margins(0)
                    plt.gca().set_aspect('equal')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.title('actual_traj')
                    plt.tight_layout()

                    # save image
                    if args.save:
                        save_path = f"./testing_log/{args.name}/images"
                        os.makedirs(save_path, exist_ok=True)
                        plt.savefig(
                            f'{save_path}/{scenario_id}_{timestep}.pdf', dpi=600)

                    # clear
                    plt.pause(0.1)
                    plt.clf()

    # save results
    df = pd.DataFrame(data={'collision': collisions, 'red_light': red_light, 'off_route': off_route,
                            'Acc': Accs, 'Jerk': Jerks, 'Lat_Acc': Lat_Accs,
                            'Human_Acc': Human_Accs, 'Human_Jerk': Human_Jerks, 'Human_Lat_Acc': Human_Lat_Accs,
                            'Prediction_ADE': prediction_ADE, 'Prediction_FDE_1': prediction_FDE_1, 'Prediction_FDE_3': prediction_FDE_3, 'Prediction_FDE_5': prediction_FDE_5,
                            'Human_L2_1s': similarity_1s, 'Human_L2_3s': similarity_3s, 'Human_L2_5s': similarity_5s})
    df.to_csv(f'./testing_log/{args.name}/testing_log.csv')


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Open_loop_testing')
    parser.add_argument('--name', type=str, help='log name',
                        default="OpenLoopTest1")
    parser.add_argument('--open_loop_test_set', type=str,
                        help='path to testing datasets')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true",
                        help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--model_type', type=str,
                        help='DIPP or VCDI or Gaussian', default='VCDI')
    parser.add_argument('--render', action="store_true",
                        help='if render the scenario (default: False)', default=False)
    parser.add_argument('--save', action="store_true",
                        help='if save the rendered images (default: False)', default=False)
    parser.add_argument(
        '--device', type=str, help='run on which device (default: cpu)', default='cpu')
    args = parser.parse_args()

    # Run
    open_loop_test()
