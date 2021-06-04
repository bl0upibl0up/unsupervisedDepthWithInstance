# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import ModelOptions
from datasets import KITTIOdomDataset
import networks


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    encoder_path = os.path.join(opt.load_weights_folder, 'encoder.pth')
    pose_decoder_path = os.path.join(opt.load_weights_folder, 'pose.pth')

    pose_att_enc_path = []
    pose_att_enc = []
    attention_channels_enc = np.array([64, 128, 256, 512])
    
    
    for i in range(4):
        pose_att_enc_path.append(os.path.join(opt.load_weights_folder, '(\'pose_att_enc\', ' + str(i) + ').pth'))
        
        if i == 0:
            pose_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i],
                                                                                  attention_channels_enc[i], do_pool=True))
        else:
            pose_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i] + attention_channels_enc[i-1],
                                                                                 attention_channels_enc[i], do_pool=False))
        pose_att_enc[i].load_state_dict(torch.load(pose_att_enc_path[i]))

    encoder_dict = torch.load(encoder_path)

    encoder = networks.ResnetEncoderMtan(opt.num_layers, False)
    pose_decoder = networks.PoseDecoder(encoder.num_ch_enc, 2)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k,v in encoder_dict.items() if k in model_dict})
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    encoder.cuda()
    encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    for i in range(4):
        pose_att_enc[i].cuda()
        pose_att_enc[i].eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids])
            o1, l11, l12, l21, l22, l31, l32, l41, l42 = encoder(all_color_aug)
            features = [o1, l12, l22, l32, l42]
            nested_features = [l11, l21, l31, l41]

            pose_enc_outputs = []

            for i in range(4):
                if i == 0:
                    attention_inputs = [nested_features[i], features[i+1]]
                else:
                    attention_inputs = [torch.cat([pose_enc_outputs[i-1], nested_features[i]], dim=1), features[i+1]]
                pose_enc_outputs.append(pose_att_enc[i](attention_inputs))

            p_features = torch.split(pose_enc_outputs[-1], opt.batch_size)
            pose_features = {}
            for i, k in enumerate(opt.frame_ids):
                pose_features[k] = p_features[i]
            pose_inputs = [[pose_features[0]], [pose_features[1]]]
            axisangle, translation = pose_decoder(pose_inputs)

            #features = [encoder(all_color_aug)]
            #axisangle, translation = pose_decoder(features)

            pred_poses.append(
               transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = ModelOptions()
    evaluate(options.parse())
