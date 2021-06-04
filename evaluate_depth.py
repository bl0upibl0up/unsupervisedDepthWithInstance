from __future__ import absolute_import, division, print_function

import os 
import cv2
import numpy as np

import torch 
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import ModelOptions
import datasets
import networks
import matplotlib.pyplot as plt
from scipy import stats as scpystats

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), 'splits')

STEREO_SCALE_FACTOR = 5.4

baseline = 0.22
focal = 2262


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_log = np.mean(np.abs((np.log(gt) - np.log(pred))))


    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    log_diff = np.log(gt) - np.log(pred)
    num_pixels = float(log_diff.size)
    scale_inv = np.sqrt(np.sum(np.square(log_diff))/num_pixels - np.square(np.sum(log_diff))/np.square(num_pixels))

    kl_div = scpystats.entropy(gt, pred)


    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, scale_inv, abs_log, kl_div

def evaluate(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
            'Choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo'
    if opt.ext_disp_to_eval is None:
        encoder_path = os.path.join(opt.load_weights_folder, 'encoder.pth')
        decoder_path = os.path.join(opt.load_weights_folder, 'decoder.pth')
        depth_att_enc_path = []
        depth_att_dec_path = []
        depth_att_enc = []
        depth_att_dec = []

        attention_channels_enc = np.array([64, 128, 256, 512])
        attention_channels_dec = np.array([16, 32, 64, 128, 256, 512])

        for i in range(5):
            if i < 4:
                depth_att_enc_path.append(os.path.join(opt.load_weights_folder, '(\'depth_att_enc\', ' + str(i) + ').pth'))
                if i == 0:
                    depth_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i],
                                                                                  attention_channels_enc[i], do_pool=True))
                else:
                    depth_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i] + attention_channels_enc[i-1],
                                                                                 attention_channels_enc[i], do_pool=False))
                depth_att_enc[i].load_state_dict(torch.load(depth_att_enc_path[i]))


            depth_att_dec_path.append(os.path.join(opt.load_weights_folder, '(\'depth_att_dec\', ' + str(i) + ').pth'))
            if i < 2:
                depth_att_dec.append(networks.AttentionDecoder(attention_channels_dec[-i-1], attention_channels_dec[-i-2], do_upsample=False))
            else:
                depth_att_dec.append(networks.AttentionDecoder(attention_channels_dec[-i-1], attention_channels_dec[-i-2], do_upsample=True))
            depth_att_dec[i].load_state_dict(torch.load(depth_att_dec_path[i]))





        att_disp_3_path = os.path.join(opt.load_weights_folder, '(\'att_disp\', 0).pth')
        att_disp_3 = networks.AttentionDispDecoder(0)
        att_disp_3.load_state_dict(torch.load(att_disp_3_path))
        print('splits_dir = ', splits_dir)
        print('eval_split = ', opt.eval_split)
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, 'test_files.txt'))

        encoder_dict = torch.load(encoder_path)

        if opt.dataset == 'kitti':
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
        elif opt.dataset == 'cityscapes':
            dataset = datasets.cityscapes_dataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers = opt.num_workers, pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoderMtan(opt.num_layers, False)
        decoder = networks.SharedDecoder(encoder.num_ch_enc, num_output_channels=2)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k,v in encoder_dict.items() if k in model_dict})
        decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        decoder.cuda()
        decoder.eval()

        for i in range(5):
            if i < 4:
                depth_att_enc[i].cuda()
                depth_att_enc[i].eval()
            depth_att_dec[i].cuda()
            depth_att_dec[i].eval()

        att_disp_3.cuda()
        att_disp_3.eval()

        pred_disps = []
        gt_depths = []
        print('-> Computing predictions with size {}x{}'.format(encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                if opt.dataset == 'cityscapes':
                    gt_depths.append(data['depth_gt'].squeeze())
                input_color = data[('color', 0, 0)].cuda()
                o1, l11, l12, l21, l22, l31, l32, l41, l42 = encoder(input_color)
                features = [o1, l12, l22, l32, l42]
                nested_features = [l11, l21, l31, l41]

                shared_features_decoder, _ = decoder(features)

                depth_enc_outputs = []
                depth_dec_outputs = []
                for i in range(4):
                    if i == 0:
                        attention_inputs = [nested_features[i], features[i+1]]
                    else:
                        attention_inputs = [torch.cat([depth_enc_outputs[i-1], nested_features[i]],dim=1), features[i+1]]
                    temp_out = depth_att_enc[i](attention_inputs)
                    depth_enc_outputs.append(temp_out)
                for i in range(5):
                    k = 4 - i
                    if i == 0:
                        attention_inputs = [depth_enc_outputs[k-1], shared_features_decoder[('upconv', k, 0)],
                                            shared_features_decoder[('upconv', k, 1)]]
                    else:
                        attention_inputs = [depth_dec_outputs[i-1], shared_features_decoder[('upconv', k, 0)],
                                            shared_features_decoder[('upconv', k, 1)]]
                    temp_out = depth_att_dec[i](attention_inputs)
                    depth_dec_outputs.append(temp_out)
                    if i == 4:
                        disp = att_disp_3(depth_dec_outputs[i])
                        disp = disp.squeeze()
                        disp = disp.cpu()
                        pred_disps.append(disp)

        pred_disps = np.concatenate(pred_disps)
        print(pred_disps.shape)
        if opt.dataset == 'cityscapes':
            gt_depths = np.concatenate(gt_depths)
        del attention_inputs
        del depth_dec_outputs
        del features
        del nested_features
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_dir = opt.output_dir + opt.save_pred_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(
            output_dir, "disps_{}.npy".format(opt.save_pred_name))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    if opt.dataset == 'kitti':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    del dataset
    del dataloader
    del encoder
    del depth_att_enc_path
    del depth_att_dec_path
    del depth_att_enc
    del depth_att_dec
    

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    while pred_disps.shape[0] >= 1:
        gt_depth = gt_depths[0]
        gt_depths = np.delete(gt_depths, 0, 0)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[0]
        pred_disps = np.delete(pred_disps, 0, 0)
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_disp = 1 / pred_disp
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_disp = pred_disp[mask]
        gt_depth = gt_depth[mask]

        pred_disp *= opt.pred_depth_scale_factor


        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_disp)
            ratios.append(ratio)
            pred_disp *= ratio

        pred_disp[pred_disp < MIN_DEPTH] = MIN_DEPTH
        pred_disp[pred_disp > MAX_DEPTH] = MAX_DEPTH
        
        errors.append(compute_errors(gt_depth, pred_disp))
    output_dir = opt.output_dir + opt.save_pred_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir, "errors_{}.npy".format(opt.save_pred_name))
    np.save(output_path, errors)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 10).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_inv", "abs_log", "kl_div"))
    print(("|{: 8.3f}  " * 10).format(*mean_errors.tolist()))
    
    print("\n-> Done!")


if __name__ == '__main__':
    options = ModelOptions()
    evaluate(options.parse())
