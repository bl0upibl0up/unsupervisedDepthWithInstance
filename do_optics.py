from OPTICS import *
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import argparse
import os 
import cv2

cv2.setNumThreads(0)


parser = argparse.ArgumentParser(description='do optics')

parser.add_argument('--results_path')
parser.add_argument('--dataset')

args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'cityscapes':
        files_1 = os.listdir(args.results_path)
        files_1 = [f for f in files_1 if f.endswith('npy')]
        
        results_path = args.results_path.replace('part1', 'part2')
        files_2 = os.listdir(results_path)
        files_2 = [f for f in files_2 if f.endswith('npy')]

        ins_files_1 = [f for f in files_1 if 'errors' not in f]
        ins_files_2 = [f for f in files_2 if 'errors' not in f]

        path_ins_1 = os.path.join(args.results_path, ins_files_1[0])
        path_ins_2 = os.path.join(results_path, ins_files_2[0])

        ins_1 = np.load(path_ins_1)
        ins_2 = np.load(path_ins_2)

        ins = np.concatenate((ins_1, ins_2))
    else:
        files = os.listdir(args.results_path)
        files = [f for f in files if f.endswith('npy')]

        ins_files = [f for f in files if 'ins' in f]
        
        path_ins_1 = os.path.join(args.results_path, ins_files[0])
                
        ins = np.load(path_ins_1)

    id = 0
    min_cluster_size = 20
    epsilon = 5

    for i in ins: 
        if id >= 0:
            x = i[0]
            y = i[1]
            x[x>1] = 1.0
            y[y>1] = 1.0
            x[x<0] = 0.0
            y[y<0] = 0.0

            mask = np.zeros(x.shape)
            mask[np.where(x != 0)] = 1
            xymask = np.zeros([x.shape[0], x.shape[1], 3])

            xymask[:,:,0] = y
            xymask[:,:,1] = x
            xymask[:,:,2] = mask
            
            pts_list = pre_process(xymask)
            op = Optics(pts_list, min_cluster_size, epsilon)
            order = op.run()
            clusters = op.cluster(1)

            new_mask = np.zeros(x.shape)
            for i, cluster in zip(range(len(clusters)), clusters):
                for pt in cluster.points:
                    new_mask[pt._row, pt._col] = (i+1)
            
            new_mask = to_rgb(new_mask)
            new_mask = np.uint8(new_mask)
            print(new_mask.shape)
            im = Image.fromarray(new_mask)

            
            
            img_dir = os.path.join(args.results_path, 'instance_masks')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            mask_path = os.path.join(img_dir, '{:04d}_mask.png'.format(id))
            print(mask_path)

            im.save(mask_path)
        id += 1


