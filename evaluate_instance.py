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

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), 'splits')


def compute_errors(gt, pred, mask):
    """Computation of error metrics between predicted and ground truth depths
    """
    mse = (gt - pred)**2
    mse = np.mean(mse)
    x_gt = gt[0,:,:]
    y_gt = gt[1,:,:]
    x_gt_m = x_gt[mask]
    y_gt_m = y_gt[mask]
    gt_m = np.stack((x_gt_m, y_gt_m))

    x_pred = pred[0,:,:]
    y_pred = pred[1,:,:]
    x_pred_m = x_pred[mask]
    y_pred_m = y_pred[mask]
    pred_m = np.stack((x_pred_m, y_pred_m))

    mse_masked = (gt_m - pred_m)**2
    mse_masked = np.mean(mse_masked)
    return mse, mse_masked

def evaluate(opt):
    encoder_path = os.path.join(opt.load_weights_folder, 'encoder.pth')
    decoder_path = os.path.join(opt.load_weights_folder, 'decoder.pth')

    ins_att_enc_path = []
    ins_att_dec_path = []
    ins_att_enc = []
    ins_att_dec = []

    attention_channels_enc = np.array([64, 128, 256, 512])
    attention_channels_dec = np.array([16, 32, 64, 128, 256, 512])

    for i in range(5):
        if i < 4:
            ins_att_enc_path.append(os.path.join(opt.load_weights_folder, '(\'instance_att_enc\', ' + str(i) + ').pth'))
            if i == 0:
                ins_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i],
                                                             attention_channels_enc[i], do_pool=True))
            else:
                ins_att_enc.append(networks.AttentionEncoder(attention_channels_enc[i] + attention_channels_enc[i-1],
                                                             attention_channels_enc[i], do_pool=False))
            ins_att_enc[i].load_state_dict(torch.load(ins_att_enc_path[i]))

        ins_att_dec_path.append(os.path.join(opt.load_weights_folder, '(\'instance_att_dec\', ' + str(i) + ').pth'))
        if i < 2:
            ins_att_dec.append(networks.AttentionDecoder(attention_channels_dec[-i-1], attention_channels_dec[-i-2], do_upsample=False))
        else:
            ins_att_dec.append(networks.AttentionDecoder(attention_channels_dec[-i-1], attention_channels_dec[-i-2], do_upsample=True))
        ins_att_dec[i].load_state_dict(torch.load(ins_att_dec_path[i]))

    att_ins_0_path = os.path.join(opt.load_weights_folder, '(\'att_ins\', 0).pth')
    att_ins_0 = networks.AttentionInsDecoder(0, num_output_channels=2)
    att_ins_0.load_state_dict(torch.load(att_ins_0_path))


    filenames = readlines(os.path.join(splits_dir, opt.eval_split, 'test_files.txt'))
    
    encoder_dict = torch.load(encoder_path)

    if opt.dataset == 'kitti':
        print('The model cannot be evaluated on kitti')
    elif opt.dataset == 'cityscapes':
        dataset = datasets.cityscapes_dataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
                                                [0], 4, is_train=False)
                                                
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoderMtan(opt.num_layers, False)
    decoder = networks.SharedDecoder(encoder.num_ch_enc, num_output_channels=2)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    decoder.cuda()
    decoder.eval()

    for i in range(5):
        if i < 4:
            ins_att_enc[i].cuda()
            ins_att_enc[i].eval()
        ins_att_dec[i].cuda()
        ins_att_dec[i].eval()

    att_ins_0.cuda()
    att_ins_0.eval()

    pred_ins = []
    instance_mask_gt = []

    print('-> Computing predictions with size {}x{}'.format(encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in dataloader:
            input_color = data[('color', 0, 0)].cuda()
            temp = [data[('instance_mask_x', 0, 0)], data[('instance_mask_y', 0, 0)]]
            instance_mask_gt.append(np.concatenate(temp, axis=1))
            o1, l11, l12, l21, l22, l31, l32, l41, l42 = encoder(input_color) #because I messed up with my encoder
            features = [o1, l12, l22, l32, l42]
            nested_features = [l11, l21, l31, l41]


            shared_features_decoder, _ = decoder(features)

            ins_enc_outputs = []
            ins_dec_outputs = []

            for i in range(4):
                if i == 0:
                    attention_inputs = [nested_features[i], features[i+1]]
                else:
                    attention_inputs = [torch.cat([ins_enc_outputs[i-1], nested_features[i]], dim=1), features[i+1]]
                ins_enc_outputs.append(ins_att_enc[i](attention_inputs))

            for i in range(5):
                k = 4 - i
                if i == 0:
                    attention_inputs = [ins_enc_outputs[k-1], shared_features_decoder[('upconv', k, 0)],
                                        shared_features_decoder[('upconv', k, 1)]]
                else:
                    attention_inputs = [ins_dec_outputs[i - 1], shared_features_decoder[('upconv', k, 0)],
                                        shared_features_decoder[('upconv', k, 1)]]
                ins_dec_outputs.append(ins_att_dec[i](attention_inputs))
                if i == 4:
                    ins = att_ins_0(ins_dec_outputs[i])
                    ins = ins.squeeze()
                    pred_ins.append(ins)



    pred_ins = np.concatenate(pred_ins)
    instance_mask_gt = np.concatenate(instance_mask_gt)
    print(pred_ins.shape)
   
    if opt.save_pred_instances:
        output_dir = opt.output_dir + opt.save_pred_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(
            output_dir, "ins_{}.npy".format(opt.save_pred_name))
        print("-> Saving predicted instances to ", output_path)
        np.save(output_path, pred_ins)

    errors = []
    while pred_ins.shape[0] >= 1:
        pred = pred_ins[0]
        pred_ins = np.delete(pred_ins, 0, 0)

        gt = instance_mask_gt[0]
        x = gt[0,:,:]
        mask = np.zeros(x.shape)

        mask[x != 1] = 1
        mask = mask.astype(np.uint8)
        instance_mask_gt = np.delete(instance_mask_gt, 0, 0)

        errors.append(compute_errors(gt, pred, mask))


    output_dir = opt.output_dir + opt.save_pred_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir, "errors_{}.npy".format(opt.save_pred_name))
    
    np.save(output_path, errors)
        
    mean_errors = np.array(errors).mean(0)

    

    print('\n' + ("{:>8} | " * 2).format('mse', 'mse_masked'))
    print(('|{: 8.3f}' * 2).format(*mean_errors.tolist()))
        
if __name__ == '__main__':
    options = ModelOptions()
    evaluate(options.parse())
