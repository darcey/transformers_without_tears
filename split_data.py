import argparse
import sys
import shutil
import os
import math
import all_constants as ac

# Takes a directory containing the original data, and splits it up into different versions,
# with the source side data truncated different amounts.
# To be used with preprocessing_splits in Darcey's experiments.

# The original data directory should have a structure like this:

# de_en
#   dev.de
#   dev.en
#   test.de
#   test.en
#   train.de
#   train.en

# You would pass in the de_en directory here (not its parent directory as with the original scripts).

# The truncated data directory will end up looking like this:

# de_en_trunc
#   de0_en
#       dev.de0
#       dev.en
#       test.de0
#       test.en
#       train.de0
#       train.en
#   de10_en
#       ...
#   ...
#   de100_en
#       ...

# Here you would pass in de_en_trunc as the argument.

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-data-dir', type=str, required=True,
                        help='directory containing the original (un-truncated) data')
    parser.add_argument('--trunc-data-dir', type=str, required=True,
                        help='directory to put the new, truncated data')
    parser.add_argument('--source', type=str, required=True,
                        help='source language, e.g. de')
    parser.add_argument('--target', type=str, required=True,
                        help='target language, e.g. en')
    parser.add_argument('--split-increment', type=int, required=True,
                        help='percent increment for the source side splits (e.g. 5 for 5%)')
    return parser

if __name__ == '__main__':

    # get and process args
    args = get_parser().parse_args()
    orig_dir = args.orig_data_dir
    trunc_dir = args.trunc_data_dir
    src = args.source
    tgt = args.target
    inc = args.split_increment

    # make the directories for the splits
    if not os.path.exists(trunc_dir):
        os.makedirs(trunc_dir)
    dirs = {}
    for i in range(0,101,inc):
        dir_i = os.path.join(trunc_dir, f'{src}{i}_{tgt}')
        dirs[i] = dir_i
        os.mkdir(dir_i)

    # copy all the target data over
    for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
        orig_file = os.path.join(orig_dir, f'{mode}.{tgt}')
        for i in range(0,101,inc):
            shutil.copy(orig_file, dirs[i])

    # read in the source data, do the splits
    for mode in [ac.TRAIN, ac.DEV, ac.TEST]:
        orig_file = os.path.join(orig_dir, f'{mode}.{src}')
        with open(orig_file) as f:
            orig_data = f.readlines()
        for i in range(0,101,inc):
            new_file = os.path.join(dirs[i], f'{mode}.{src}{i}')
            with open(new_file, 'w') as f:
                for orig_line in orig_data:
                    toks = orig_line.split()
                    percent = float(i)/100
                    keep = math.ceil(percent * len(toks))
                    new_line = ' '.join(toks[:keep])
                    f.write(new_line + '\n')
