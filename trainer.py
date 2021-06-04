from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import json

from layers import *
from utils import *
from kitti_utils import *

import datasets
import networks





class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.count_parameters = 0

        self.lambdas = torch.ones(2, dtype=torch.float, requires_grad = False)
        self.avg_costs = np.zeros([self.opt.num_epochs, 2], dtype=np.float32)

        self.temperature = self.opt.temperature

        self.device = torch.device('cuda')

        self.lambdas = self.lambdas.to(self.device)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == 'pairs' else self.num_input_frames
        self.attention_channels_enc = np.array([64, 128, 256, 512])
        self.attention_channels_dec = np.array([16, 32, 64, 128, 256, 512])

        self.models['encoder'] = networks.ResnetEncoderMtan(self.opt.num_layers, self.opt.weights_init == 'pretrained')
        self.models['decoder'] = networks.SharedDecoder(self.models['encoder'].num_ch_enc, num_output_channels=2)

        self.parameters_to_train += list(self.models['encoder'].parameters())
        self.parameters_to_train += list(self.models['decoder'].parameters())

        self.count_parameters += sum(p.numel() for p in self.models['decoder'].parameters() if p.requires_grad)
        self.count_parameters += sum(p.numel() for p in self.models['encoder'].parameters() if p.requires_grad)

        if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
            print('on multiple gpu')
            self.models['encoder'] = nn.DataParallel(self.models['encoder'])
            self.models['decoder'] = nn.DataParallel(self.models['decoder'])
        self.models['encoder'].to(self.device)
        self.models['decoder'].to(self.device)
        
        if self.opt.use_stereo:
            self.opt.frame_ids.append('s')


        for i in self.opt.scales:
            if 'depth' in self.opt.tasks:
                self.models[('att_disp', i)] = networks.AttentionDispDecoder(i)
                if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                    self.models[('att_disp', i)] = nn.DataParallel(self.models[('att_disp', i)])
                self.models[('att_disp', i)].to(self.device)
                self.parameters_to_train += list(self.models[('att_disp', i)].parameters())
                self.count_parameters += sum(p.numel() for p in self.models[('att_disp', i)].parameters() if p.requires_grad)
            if i == 0 and 'instance' in self.opt.tasks:
                self.models[('att_ins', i)] = networks.AttentionInsDecoder(i, num_output_channels=2)
                if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                    self.models[('att_ins', i)] = nn.DataParallel(self.models[('att_ins', i)])
                self.models[('att_ins', i)].to(self.device)
                self.parameters_to_train += list(self.models[('att_ins', i)].parameters())
                self.count_parameters += sum(p.numel() for p in self.models[('att_ins', i)].parameters() if p.requires_grad)

        for i in self.opt.tasks:

                for j in range(4):
                    if j == 0:
                        self.models[(i + '_att_enc', j)] = networks.AttentionEncoder(self.attention_channels_enc[j],
                                                                                     self.attention_channels_enc[j], do_pool=True)
                    else:
                        self.models[(i + '_att_enc', j)] = networks.AttentionEncoder(self.attention_channels_enc[j] + self.attention_channels_enc[j-1],
                                                                                     self.attention_channels_enc[j], do_pool=False)
                    if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                        self.models[(i + '_att_enc', j)] = nn.DataParallel(self.models[(i + '_att_enc', j)])
                    self.models[(i + '_att_enc', j)].to(self.device)
                    self.parameters_to_train += list(self.models[(i + '_att_enc', j)].parameters())
                    self.count_parameters += sum(p.numel() for p in self.models[(i + '_att_enc', j)].parameters() if p.requires_grad)
                if i != 'pose':
                    for j in range(5):
                        if j < 2:
                            self.models[(i + '_att_dec', j)] = networks.AttentionDecoder(
                                self.attention_channels_dec[-j-1],
                                self.attention_channels_dec[-j-2],
                                do_upsample=False)
                            
                        else:
                            self.models[(i + '_att_dec', j)] = networks.AttentionDecoder(
                                self.attention_channels_dec[-j - 1],
                                self.attention_channels_dec[-j - 2],
                                do_upsample=True)
                        if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                            self.models[(i + '_att_dec', j)] = nn.DataParallel(self.models[(i + '_att_dec', j)])
                        self.models[(i + '_att_dec', j)].to(self.device)
                        self.parameters_to_train += list(self.models[(i + '_att_dec', j)].parameters())
                        self.count_parameters += sum(p.numel() for p in self.models[(i + '_att_dec', j)].parameters() if p.requires_grad)
        if 'pose' in self.opt.tasks:

            if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                self.models['pose'] = networks.PoseDecoder(self.models['encoder'].module.num_ch_enc, self.num_pose_frames)
                self.models['pose'] = nn.DataParallel(self.models[('pose')])
            else:
                self.models['pose'] = networks.PoseDecoder(self.models['encoder'].num_ch_enc, self.num_pose_frames)
            self.models['pose'].to(self.device)
            self.parameters_to_train += list(self.models['pose'].parameters())
            self.count_parameters += sum(p.numel() for p in self.models['pose'].parameters() if p.requires_grad)
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        print('number of trainable parameters: \n', self.count_parameters)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes": datasets.cityscapes_dataset,
                         'cityscapes_instance': datasets.cityscapes_dataset_instance
                         }

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.num_train_samples = num_train_samples

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            if self.epoch == 0 or self.epoch == 1:
                self.lambdas[:] = 1.0
            else:
                w_1 = self.avg_costs[self.epoch - 1, 0] / self.avg_costs[self.epoch - 2, 0]
                w_2 = self.avg_costs[self.epoch - 1, 1] / self.avg_costs[self.epoch - 2, 1]
                self.lambdas[0] = 2 * np.exp(w_1 / self.opt.temperature) / (np.exp(w_1 / self.opt.temperature) + np.exp(w_2 / self.opt.temperature))
                self.lambdas[1] = 2 * np.exp(w_2 / self.opt.temperature) / (np.exp(w_1 / self.opt.temperature) + np.exp(w_2 / self.opt.temperature))
            self.run_epoch()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        self.model_lr_scheduler.step()

        print('Training')
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            self.model_optimizer.zero_grad()

            outputs, losses = self.process_batch(inputs, batch_idx)

            if self.opt.weighting == 'dwa':
                if 'instance' in self.opt.tasks and 'depth' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = torch.mean(self.lambdas[0].item()*losses['instance_loss'] + self.lambdas[1].item()*losses['depth_loss'])
                    else:
                        loss = self.lambdas[1].item()*losses['depth_loss']
                elif 'instance' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = losses['instance_loss']
                    else: 
                        loss = 0
                elif 'depth' in self.opt.tasks:
                    loss = losses['depth_loss']
            elif self.opt.weighting == 'kendall':
                if 'instance' in self.opt.tasks and 'depth' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = 1/2*torch.exp(-outputs['logsigma'][0])*losses['instance_loss'] + outputs['logsigma'][0]/2 + 1/2*torch.exp(-outputs['logsigma'][1])*losses['depth_loss'] + outputs['logsigma'][1]/2
                    else:
                        loss = 1/2*torch.exp(-outputs['logsigma'][1])*losses['depth_loss'] + outputs['logsigma'][1]/2
                elif 'instance' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = losses['instance_loss']
                    else:
                        loss = 0
                elif 'depth' in self.opt.tasks:
                    loss = losses['depth_loss']
            elif self.opt.weighting == 'equal':
                if 'instance' in self.opt.tasks and 'depth' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = 0.5*(losses['instance_loss'] + losses['depth_loss'])
                    else:
                        loss = losses['depth_loss']
                elif 'instance' in self.opt.tasks:
                    if losses['instance_loss'] is not None:
                        loss = losses['instance_loss']
                    else:
                        loss = torch.zeros(1, requires_grad=True)
                elif 'depth' in self.opt.tasks:
                    loss = losses['depth_loss']

            loss.backward()

            self.model_optimizer.step()


            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss.cpu().data)
                self.log('train', inputs, outputs, losses)

                self.val(batch_idx)
            self.step += 1

    def process_batch(self, inputs, batch_idx):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        all_color_aug = torch.cat([inputs['color_aug', i, 0] for i in self.opt.frame_ids])
        o1, l11, l12, l21, l22, l31, l32, l41, l42 = self.models['encoder'](all_color_aug)
        shared_features_encoder = [o1, l12, l22, l32, l42]
        nested_features_encoder = [l11, l21, l31, l41]
        shared_features_encoder = [torch.split(f, self.opt.batch_size) for f in shared_features_encoder]
        nested_features_encoder = [torch.split(f, self.opt.batch_size) for f in nested_features_encoder]

        features = {}
        nested_features = {}

        for h in self.opt.tasks:
            for i, k in enumerate(self.opt.frame_ids):
                features[(h, k)] = [f[i] for f in shared_features_encoder]
                nested_features[(h, k)] = [f[i] for f in nested_features_encoder]

        if 'depth' in self.opt.tasks:
            shared_features_decoder, outputs['logsigma'] = self.models['decoder'](features[('depth',0)])
        else:
            shared_features_decoder, outputs['logsigma'] = self.models['decoder'](features[('instance', 0)])


        attention_inputs = {}
        for i in self.opt.tasks:
            if i == 'pose':
                for j in range(4):
                    if j == 0 :
                        attention_inputs[(i + '_att_enc', j)] = [torch.cat([nested_features[(i, 0)][j], nested_features[(i,-1)][j], nested_features[(i,1)][j]], dim=0),
                                                    torch.cat([features[(i,0)][j+1], features[(i,-1)][j+1], features[(i,1)][j+1]])]
                        
                    elif j != 0:
                        attention_inputs[(i + '_att_enc', j)] = [torch.cat([outputs[(i + '_att_enc', j-1)], torch.cat([nested_features[(i, 0)][j], nested_features[(i,-1)][j], nested_features[(i,1)][j]])], dim=1),
                                                    torch.cat([features[(i,0)][j+1], features[(i,-1)][j+1], features[(i,1)][j+1]])]
                    outputs[(i + '_att_enc', j)] = self.models[(i + '_att_enc', j)](attention_inputs[(i + '_att_enc', j)])

                    
            else:
                for j in range(4):
                    if j == 0:
                        attention_inputs[(i + '_att_enc', j)] = [nested_features[(i,0)][j], features[(i,0)][j+1]]
                    elif j != 0:
                        attention_inputs[(i + '_att_enc', j)] = [torch.cat([outputs[(i + '_att_enc', j-1)], nested_features[(i,0)][j]], dim=1),
                                                    features[(i,0)][j+1]]
                    outputs[(i + '_att_enc', j)] = self.models[(i + '_att_enc', j)](attention_inputs[(i + '_att_enc', j)])
                for j in range(5):
                    k = 4-j
                    if j == 0:
                        attention_inputs[(i + '_att_dec', j)] = [outputs[(i + '_att_enc', k-1)], shared_features_decoder[('upconv', k, 0)], shared_features_decoder[('upconv', k, 1)]]
                    else:
                        attention_inputs[(i + '_att_dec', j)] = [outputs[(i + '_att_dec', j-1)], shared_features_decoder[('upconv', k, 0)], shared_features_decoder[('upconv', k, 1)]]
                    outputs[(i + '_att_dec', j)] = self.models[(i + '_att_dec', j)](attention_inputs[(i + '_att_dec', j)])
                    if j > 0:
                        if i == 'depth':
                            outputs[('att_disp', j - 1)] = self.models[('att_disp', k)](outputs[(i + '_att_dec', j)])
                    if j == 4:
                        if i == 'instance':
                            outputs[('att_ins', j - 1)] = self.models[('att_ins', k)](outputs[(i + '_att_dec', j)])
        if 'pose' in self.opt.tasks:
            outputs.update(self.predict_poses(inputs, outputs[('pose_att_enc', 3)]))
        if 'depth' in self.opt.tasks:
            self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, batch_idx)
        return outputs, losses

    def predict_poses(self, inputs, features):

        outputs ={}
        features = torch.split(features, self.opt.batch_size)
        pose_features = {}
        for i, k in enumerate(self.opt.frame_ids):
            if k != 's':
                pose_features[k] = features[i]
        for f_i in self.opt.frame_ids[1:]:
            if f_i != 's':
                if f_i < 0:
                    pose_inputs = [[pose_features[f_i]], [pose_features[0]]]
                else:
                    pose_inputs = [[pose_features[0]], [pose_features[f_i]]]

                axisangle, translation = self.models['pose'](pose_inputs)
                outputs[('axisangle', 0, f_i)] = axisangle
                outputs[('translation', 0, f_i)] = translation
                outputs[('cam_T_cam', 0, f_i)] = transformation_from_parameters(
                    axisangle[:,0], translation[:,0], invert=(f_i < 0))

        return outputs

    def generate_images_pred(self, inputs, outputs):

        for scale in self.opt.scales:
            disp = outputs[('att_disp', scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode='bilinear', align_corners = False)
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[('depth', 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == 's':
                    T = inputs['stereo_T']
                else:
                    T = outputs[('cam_T_cam', 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target.to(self.device)).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, batch_idx):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_loss_scales = 0

        if 'instance' in self.opt.tasks and 'cityscapes' in self.opt.dataset:
            labels = [inputs[('instance_mask_x', 0, 0)], inputs[('instance_mask_y', 0, 0)]]
            labels = torch.cat(labels, 1)
            labels = labels.to(self.device)

            ins = outputs[('att_ins', 3)]
            ins_cost = self.l1_loss_instance(ins, labels)
            if ins_cost is not None:
                total_loss += ins_cost
                self.avg_costs[self.epoch, 0] += ins_cost.detach().item()/self.num_train_samples
            else:
                total_loss += 0
        if 'depth' in self.opt.tasks:
            for scale in self.opt.scales:
                loss = 0
                reprojection_losses = []

                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0
                
                disp = outputs[('att_disp', 3 - scale)]
                disp = torch.split(disp, self.opt.batch_size)
                disp = disp[0]
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                reprojection_losses = torch.cat(reprojection_losses, 1)
                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                elif self.opt.predictive_mask:
                    # use the predicted mask
                    mask = outputs["predictive_mask"]["disp", scale]
                    if not self.opt.v1_multiscale:
                        mask = F.interpolate(
                            mask, [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)

                    reprojection_losses *= mask

                    # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                    weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                    loss += weighting_loss.mean()

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).cuda() * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = (idxs > 1).float()

                loss += to_optimise.mean()
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
                disp_smooth_cost = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                loss += disp_smooth_cost

                total_loss_scales += loss
            total_loss += total_loss_scales/self.num_scales
            self.avg_costs[self.epoch, 1] += total_loss_scales.detach().item()/self.num_scales/self.num_train_samples
        if 'instance' in self.opt.tasks:
            if ins_cost is not None:
                losses['instance_loss'] = ins_cost
            else:
                losses['instance_loss'] = None
        if 'depth' in self.opt.tasks:
            losses['depth_loss'] = total_loss_scales/self.num_scales
        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())


    def val(self, batch_idx):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs, batch_idx)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def l1_loss_instance(self, input, target, val=False):
        if val:
            size_average = False
        else:
            size_average = True
        mask = target != 1
        if mask.data.sum() < 1:
            print('no instance in data')
            return None
        #
        abs_diff = torch.abs(target[mask] - input[mask])
        lss = abs_diff.mean()
        return lss

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if v is not None:
                writer.add_scalar("{}".format(l), v, self.step)

        writer.add_scalar('lambda_ins', self.lambdas[0], self.step)
        writer.add_scalar('lambda_disp', self.lambdas[1], self.step)
        for i in range(2):
            writer.add_scalar('logsigma_{}'.format(i), outputs['logsigma'][i], self.step)
        for j in range(3):
            writer.add_image('color_{}_{}/{}'.format(0, 0, j), inputs[('color', 0, 0)][j].data, self.step)
            if 'instance' in self.opt.tasks:
                writer.add_image('instance_x_{}_{}/{}'.format(0, 0, j),
                             normalize_image(inputs[('instance_mask_x', 0, 0)][j].data), self.step)
                ins_out = F.interpolate(outputs[('att_ins', 3)], [self.opt.height, self.opt.width], mode='bilinear', align_corners=False)
                instance_x = torch.split(ins_out, 1, dim=1)

                writer.add_image('instance_x_pred{}_{}/{}'.format(0, 0, j), normalize_image(instance_x[0][j].data),
                                self.step)
                writer.add_image('instance_y_pred{}_{}/{}'.format(0, 0, j), normalize_image(instance_x[1][j].data),
                                self.step)
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0 and 'depth' in self.opt.tasks:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                if 'depth' in self.opt.tasks:
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("att_disp", s)][j, 0]), self.step)

                    if self.opt.predictive_mask:
                        writer.add_image(
                            "predictive_mask_{}/{}".format(s, j),
                            outputs["predictive_mask"][("disp", s)][j, 0], self.step)

                    elif not self.opt.disable_automasking and 'depth' in self.opt.tasks:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j], self.step)
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        with open(self.opt.models_txt) as f:
            models_to_load  = [line.rstrip() for line in f]
        for n in models_to_load:
            name, nbr = n.split()
            print('loading {} weights...'.format(n))
            path = os.path.join(self.opt.load_weights_folder, "('{}', {}).pth".format(name, nbr))
            model_dict = self.models[(name, int(nbr))].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[(name, int(nbr))].load_state_dict(model_dict)
        
        shared_components = ['encoder', 'decoder']
        for i in shared_components:
            path = os.path.join(self.opt.load_weights_folder, '{}.pth'.format(i))
            model_dict = self.models[i].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[i].load_state_dict(model_dict)
        


        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
