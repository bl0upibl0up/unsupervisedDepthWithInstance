import os
from absl import app
from absl import flags
from absl import logging
from pathlib import Path
import random
import argparse


flags.DEFINE_string('dataset', None, 'dataset name')
flags.DEFINE_string('set', None, 'train val test')
flags.DEFINE_string('input_list', None, 'input_list')
flags.DEFINE_string('split_id', None, 'split id')
flags.DEFINE_integer('nbr_of_samples', None, 'nbr of samples')

FLAGS = flags.FLAGS

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('set')
flags.mark_flag_as_required('input_list')
flags.mark_flag_as_required('split_id')
flags.mark_flag_as_required('nbr_of_samples')

def main(_):
    dataset = FLAGS.dataset
    set_type = FLAGS.set
    input_list = FLAGS.input_list
    split_id = FLAGS.split_id

    nbr_of_samples = FLAGS.nbr_of_samples

    f = open(input_list)
    filenames = f.readlines()
    f.close()
    print('number of files: ', len(filenames))
    filenames = [f for f in filenames if '000000_left' not in f]
    print('number of files without 0: ', len(filenames))
    filenames = [f for f in filenames if '000029_left' not in f]
    print('number of files without 29: ', len(filenames))
    split_filename = dataset + '_' + set_type + '_' + '{0:03d}'.format(int(split_id))+ '.txt'
    f = open(split_filename, 'w')
    count_l = 0
    count_r = 0
    for _ in range(nbr_of_samples):
        idx = random.randint(0, len(filenames)-1)
        type, set, city, image_name = str.split(filenames[idx], '/')
        _, seq_id, frame_id, _ = str.split(image_name, '_')


        prev_frame_name = city + '_' + seq_id + '_' + prev_frame_id
        next_frame_name = city + '_' + seq_id + '_' + next_frame_id

        if random.uniform(0, 0.9) <= 0.45:
            new_line = 'l ' + set_type + ' ' + city + ' ' + seq_id + ' ' + frame_id
            count_l += 1
        else:
            new_line = 'r ' + set_type + ' ' + city + ' ' + seq_id + ' ' + frame_id
            count_r += 1
        del(filenames[idx])
        filenames = [f for f in filenames if prev_frame_name not in f]
        filenames = [f for f in filenames if next_frame_name not in f]
        f.write('%s\n' % new_line)
    f.close()


if __name__ == '__main__':
    app.run(main)