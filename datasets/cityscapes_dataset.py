
from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import PIL.Image as pil
import json
from torchvision import transforms


from kitti_utils import generate_depth_map
from cityscapes_utils import generate_depth_map_cityscapes
from .mono_dataset_v2 import mono_dataset

import matplotlib.pyplot as plt

class cityscapes_dataset_instance(mono_dataset):
    
    def __init__(self, *args, **kwargs):
        super(cityscapes_dataset_instance, self).__init__(*args, **kwargs)
        self.valid_classes = [24, 25, 26, 27, 28, 31, 32, 33]
        self.full_res_shape = (2048, 1024)
        self.K = np.array([[2262.52 / 2048, 0, 0.5, 0],
                           [0, 1096.98 / 1024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def check_depth(self):
        return False

    def get_color(self, side, set_type, city, seq_index, frame_index, do_flip):
        color = self.loader(self.get_image_path(side, set_type, city, seq_index, frame_index))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    def get_instance_mask(self, side, set_type, city, seq_index, frame_index, do_flip):
        path = self.get_mask_path(set_type, city, seq_index, frame_index)
        with open(path, 'rb') as f:
            with pil.open(f) as img:
                mask = img.convert('I')
        if do_flip:
            mask = mask.transpose(pil.FLIP_LEFT_RIGHT)
        resize_mask = transforms.Resize((self.height, self.width), pil.NEAREST)
        mask = resize_mask(mask)
        return mask

    def get_image_path(self, side, set_type, city, seq_index, frame_index):
        if side == 'l':
            side_folder = 'leftImg8bit'
            side_file = 'leftImg8bit'
        else:
            side_folder = 'rightImg8bit'
            side_file = 'rightImg8bit'
        image_name = city + '_' + seq_index + '_' + frame_index + '_' + side_file + self.img_ext
        image_path = os.path.join(self.data_path, side_folder, set_type, city, image_name)
        return image_path

    def get_mask_path(self, set_type, city, seq_index, frame_index):
        image_name = city + '_' + seq_index + '_' + frame_index + '_gtFine_instanceIds' + '.png'
        image_path = os.path.join(self.data_path, 'gtFine', set_type, city, image_name)
        return image_path
    
    def encode_instancemap(self, mask, width_or_height='width'):

        mask = np.array(mask)
        binary_mask = np.ones(mask.shape)
        binary_mask[mask < 24000] = 0
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids >= 24000]
        sh = mask.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')
        ymap = ymap/sh[0]
        xmap = xmap/sh[0]
        out_map = np.ones(sh)
        if instance_ids.shape[0] > 1:
            for instance_id in instance_ids:
                if instance_id == 0:
                    continue
                instance_indicator = (mask == instance_id)
                if width_or_height == 'width':
                    coordinate = np.mean(xmap[instance_indicator])
                    out_map[instance_indicator] = xmap[instance_indicator] - coordinate
                elif width_or_height == 'height':
                    coordinate = np.mean(ymap[instance_indicator])
                    out_map[instance_indicator] = ymap[instance_indicator] - coordinate
            out_map[binary_mask == 0] = -1.0
            out_map = pil.fromarray(out_map)
        else:
            out_map = np.zeros(ymap.shape)
            out_map = pil.fromarray(out_map)
        return out_map
        

class cityscapes_dataset(mono_dataset):

    def __init__(self, *args, **kwargs):
        super(cityscapes_dataset, self).__init__(*args, **kwargs)
        self.ignore_index = 250
        self.K = np.array([[2262.52 / 2048, 0, 0.5, 0],
                           [0, 1096.98 / 1024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.full_res_shape = (2048, 1024)

    def check_depth(self):
        _, set_type, city, seq_index, frame_index = self.filenames[0].split()
        disp_filename = city + '_' + seq_index + '_' + frame_index + '_disparity.png'
        disp_file_path = os.path.join(self.data_path, 'disparity_sequence', set_type, city,
                                     disp_filename)
        return os.path.isfile(disp_file_path)

    def get_color(self, side, set_type, city, seq_index, frame_index, do_flip):
        crop = (192, 256, 1856, 768)
        color = self.loader(self.get_image_path(side, set_type, city, seq_index, frame_index))
        color = color.crop(crop)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_instance_mask(self, side, set_type, city, seq_index, frame_index, do_flip):
        crop = (192, 256, 1856, 768)
        path = self.get_mask_path(side, set_type, city, seq_index, frame_index)
        with open(path, 'rb') as f:
            with pil.open(f) as img:
                mask = img.convert('I')
        mask = mask.crop(crop)
        if do_flip:
            mask = mask.transpose(pil.FLIP_LEFT_RIGHT)
        resize_mask = transforms.Resize((self.height, self.width), pil.NEAREST)
        mask = resize_mask(mask)
        return mask

    def get_image_path(self, side, set_type, city, seq_index, frame_index):
        if side == 'l':
            side_folder = 'leftImg8bit_sequence'
            side_file = 'leftImg8bit'
        else:
            side_folder = 'rightImg8bit_sequence'
            side_file = 'rightImg8bit'
        image_name = city + '_' + seq_index + '_' + frame_index + '_' + side_file + self.img_ext
        image_path = os.path.join(self.data_path, side_folder, set_type, city, image_name)
        return image_path

    def get_mask_path(self, side, set_type, city, seq_index, frame_index):
        if side == 'l':
            side_folder = 'leftImg8bit_instance'
            side_file = 'leftImg8bit_instance'
        else:
            side_folder = 'rightImg8bit_instance'
            side_file = 'rightImg8bit_instance'
        image_name = city + '_' + seq_index + '_' + frame_index + '_' + side_file + '.png'
        image_path = os.path.join(self.data_path, side_folder, set_type, city, image_name)
        return image_path

    def get_intrinsics(self, set_type, city, seq_index):
        cam_file_name = city + '_' + seq_index + '_000000_camera.json'
        calib_path = os.path.join(self.data_path, 'camera', set_type, city, cam_file_name)
        with open(calib_path) as calib_json:
            calib = json.load(calib_json)

        fx = calib['intrinsic']['fx']
        fy = calib['intrinsic']['fy']
        u0 = calib['intrinsic']['u0']
        v0 = calib['intrinsic']['v0']

        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    def get_depth(self, set_type, city, seq_index, frame_index, do_flip):
        cam_file_name = city + '_' + seq_index + '_000000_camera.json'
        disp_file_name = city + '_' + seq_index + '_' + frame_index + '_disparity.png'
        calib_path = os.path.join(self.data_path, 'camera', set_type, city, cam_file_name)
        disp_path = os.path.join(self.data_path, 'disparity_sequence', set_type, city, disp_file_name)

        depth_gt = generate_depth_map_cityscapes(calib_path, disp_path)

        return  depth_gt

    def encode_instancemap(self, mask, width_or_height='width'):

        mask = np.array(mask)
        binary_mask = np.ones(mask.shape)
        binary_mask[mask == 0] = 0
        instance_ids = np.unique(mask)
        sh = mask.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')
        ymap = ymap/sh[0]
        xmap = xmap/sh[0]
        out_map = np.ones(sh)
        if instance_ids.shape[0] > 1:
            for instance_id in instance_ids:
                if instance_id == 0:
                    continue
                instance_indicator = (mask == instance_id)
                if width_or_height == 'width':
                    coordinate = np.mean(xmap[instance_indicator])
                    out_map[instance_indicator] = xmap[instance_indicator] - coordinate
                elif width_or_height == 'height':
                    coordinate = np.mean(ymap[instance_indicator])
                    out_map[instance_indicator] = ymap[instance_indicator] - coordinate
            out_map[binary_mask == 0] = -1.0
            out_map = pil.fromarray(out_map)
        else:
            out_map = np.zeros(ymap.shape)
            out_map = pil.fromarray(out_map)
        return out_map







