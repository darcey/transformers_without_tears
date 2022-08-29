import argparse
import os
from os.path import join, exists
from collections import Counter
import subprocess
import numpy as np
import all_constants as ac

# This is an alternative to the original preprocessing script, for use in Darcey's experiments.
# It assumes that split_data has already been run, creating a file structure that looks like this:

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

# This script learns (either separate or joint) BPE just once, on the full version of the data,
# stored in the src100_tgt folder. It then applies this learned BPE to all the truncated versions,
# computing separate BPE vocabularies and masks and such for each. The resulting data directory looks like:

# de_en_proc
#   lang.vocab
#   vocab.joint
#   mask.de0.npy
#   mask.de10.npy
#   ...
#   mask.de100.npy
#   mask.en.npy
#   de0_en
#       dev.de0.bpe
#       dev.de0.npy
#       dev.en.bpe
#       dev.en.npy
#       test.de0.bpe
#       test.en.bpe
#       train.de0.bpe
#       train.de0.npy
#       train.en.bpe
#       train.en.npy
#   de10_en
#       ...
#   ...
#   de100_en
#       ...

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncated-data-dir', type=str, required=True,
                        help='path to directory containing the truncated versions of the raw data')
    parser.add_argument('--processed-data-dir', type=str, required=True,
                        help='path to output the BPE\'d versions of the truncated data')
    parser.add_argument('--source', type=str, required=True,
                        help='source language, e.g. de')
    parser.add_argument('--target', type=str, required=True,
                        help='target language, e.g. en')

    parser.add_argument('--fast', type=str, required=True,
                        help='path to fastBPE binary')
    parser.add_argument('--joint', choices=('True','False'), default='True',
                        help='whether to perform joint or separate BPE')
    parser.add_argument('--joint-num-ops', type=int,
                        help='if joint BPE, number of ops for joint BPE')
    parser.add_argument('--source-num-ops', type=int,
                        help='if separate BPE, number of ops for source side BPE')
    parser.add_argument('--target-num-ops', type=int,
                        help='if separate BPE, number of ops for target side BPE')

    parser.add_argument('--source-eos', choices=('True','False'), default='True',
                        help='whether to append EOS to the end of the source sentence')
    parser.add_argument('--split-increment', type=int, required=True,
                        help='percent increment for the source side splits (e.g. 5 for 5%) (should be the same one that you used for split_data.py')
    return parser

if __name__ == '__main__':

    # get and process args
    args = get_parser().parse_args()
    src = args.source
    tgt = args.target
    fast = args.fast
    eos = (args.source_eos == 'True')
    inc = args.split_increment

    joint = (args.joint == 'True')
    if joint:
        joint_num_ops = args.joint_num_ops
    else:
        src_num_ops = args.source_num_ops
        tgt_num_ops = args.target_num_ops

    # build up directory structure
    trunc_dir = args.truncated_data_dir
    proc_dir = args.processed_data_dir
    if not exists(proc_dir):
        os.makedirs(proc_dir)
    sub_dirs = {}
    for i in range(0,101,inc):
        dir_i = f'{src}{i}_{tgt}'
        sub_dirs[i] = dir_i
        proc_dir_i = join(proc_dir, dir_i)
        os.mkdir(proc_dir_i)

    # define all hardcoded filenames
    lang_vocab_file = join(proc_dir, 'lang.vocab')
    joint_vocab_file = join(proc_dir, 'vocab.joint')

    mask_files = {}
    for i in range(0,101,inc):
        mask_files[(src,i)] = join(proc_dir, f'mask.{src}{i}.npy')
    mask_files[tgt] = join(proc_dir, f'mask.{tgt}.npy')

    input_files = {}
    bpe_files = {}
    npy_files = {}
    for i in range(0,101,inc):
        for mode in ac.TRAIN, ac.DEV, ac.TEST:
            input_files[(src, mode, i)] = join(trunc_dir, sub_dirs[i], f'{mode}.{src}{i}')
            input_files[(tgt, mode, i)] = join(trunc_dir, sub_dirs[i], f'{mode}.{tgt}')
            bpe_files[(src, mode, i)] = join(proc_dir, sub_dirs[i], f'{mode}.{src}{i}.bpe')
            bpe_files[(tgt, mode, i)] = join(proc_dir, sub_dirs[i], f'{mode}.{tgt}.bpe')
            npy_files[(src, mode, i)] = join(proc_dir, sub_dirs[i], f'{mode}.{src}{i}.npy')
            npy_files[(tgt, mode, i)] = join(proc_dir, sub_dirs[i], f'{mode}.{tgt}.npy')

    joint_code_file = join(proc_dir, 'joint.bpe')
    src_code_file = join(proc_dir, f'{src}.bpe')
    tgt_code_file = join(proc_dir, f'{tgt}.bpe')

    vocab_files = {}
    for i in range(0,101,inc):
        vocab_files[(src, i)] = join(proc_dir, f'{src}{i}.vocab')
    vocab_files[tgt] = join(proc_dir, f'{tgt}.vocab')

    # save language vocab
    langs = [f'{src}{i}' for i in range(0,101,inc)] + [tgt]
    with open(lang_vocab_file, 'w') as fout:
        for idx, lang in enumerate(langs):
            fout.write(f'{lang} {idx}\n')

    # learn BPE codes just once, on the untruncated version of the data
    # (this way the BPE codes are the same across the different truncated copies,
    #  making them easier to compare to one another and reducing confounders)
    src_input = input_files[(src, ac.TRAIN, 100)]
    tgt_input = input_files[(tgt, ac.TRAIN, 100)]
    if joint:
        print("Learn joint BPE from the untruncated data")
        command = f'{fast} learnbpe {joint_num_ops} {src_input} {tgt_input} > {joint_code_file}'
        print(command)
        subprocess.check_call(command, shell=True)
    else:
        print("Learn separate BPE from the untruncated data")
        command = f'{fast} learnbpe {src_num_ops} {src_input} > {src_code_file}'
        print(command)
        subprocess.check_call(command, shell=True)

        command = f'{fast} learnbpe {tgt_num_ops} {tgt_input} > {tgt_code_file}'
        print(command)
        subprocess.check_call(command, shell=True)

    # apply BPE to the various training source and target sides
    # (this involves repeating some work because the target sides should
    #  be the same across all the splits but whatever)
    src_codes = joint_code_file if joint else src_code_file
    tgt_codes = joint_code_file if joint else tgt_code_file
    print("Apply BPE to all of the training data")
    for i in range(0,101,inc):
        src_raw = input_files[(src, ac.TRAIN, i)]
        src_bpe = bpe_files[(src, ac.TRAIN, i)]
        command = f'{fast} applybpe {src_bpe} {src_raw} {src_codes}'
        print(command)
        subprocess.check_call(command, shell=True)

        tgt_raw = input_files[(tgt, ac.TRAIN, i)]
        tgt_bpe = bpe_files[(tgt, ac.TRAIN, i)]
        command = f'{fast} applybpe {tgt_bpe} {tgt_raw} {tgt_codes}'
        print(command)
        subprocess.check_call(command, shell=True)

    # learn a vocabulary of BPE codes for each version of the source data,
    # and just once for the target data (since all target data is the same)
    print("Extract BPE vocabs from the different versions of the training data")
    for i in range(0,101,inc):
        bpe_file = bpe_files[(src, ac.TRAIN, i)]
        vocab_file = vocab_files[(src, i)]
        command = f'{fast} getvocab {bpe_file} > {vocab_file}'
        print(command)
        subprocess.check_call(command, shell=True)
    bpe_file = bpe_files[(tgt, ac.TRAIN, 100)]
    vocab_file = vocab_files[tgt]
    command = f'{fast} getvocab {bpe_file} > {vocab_file}'
    print(command)
    subprocess.check_call(command, shell=True)

    # based on these vocabs, apply BPE to dev and test data
    # (again, this will end up repeating some work, since target files
    #  are the same across all the data splits)
    print("Apply BPE to dev and test data, using vocabs learned")
    tgt_vocab_file = vocab_files[tgt]
    for i in range(0,101,inc):
        src_vocab_file = vocab_files[(src, i)]
        for mode in [ac.DEV, ac.TEST]:
            src_raw = input_files[(src, mode, i)]
            src_bpe = bpe_files[(src, mode, i)]
            command = f'{fast} applybpe {src_bpe} {src_raw} {src_codes} {src_vocab_file}'
            print(command)
            subprocess.check_call(command, shell=True)

            tgt_raw = input_files[(tgt, mode, i)]
            tgt_bpe = bpe_files[(tgt, mode, i)]
            command = f'{fast} applybpe {tgt_bpe} {tgt_raw} {tgt_codes} {tgt_vocab_file}'
            print(command)
            subprocess.check_call(command, shell=True)

    # now we extract a joint vocabulary from the full versions of the encoded training data
    # we also save the vocab for each individual data split
    # this is used to get the logit mask
    def count_toks(filename, counter):
        with open(filename, 'r') as fin:
            for line in fin:
                toks = line.strip().split()
                if toks:
                    counter.update(toks)

    joint_vocab = Counter()
    for lang in [src, tgt]:
        bpe_file = bpe_files[(lang, ac.TRAIN, 100)]
        count_toks(bpe_file, joint_vocab)
        
    sub_vocabs = {(src, i): Counter() for i in range(0,101,inc)}
    for i in range(0,101,inc):
        bpe_file = bpe_files[(src, ac.TRAIN, i)]
        count_toks(bpe_file, sub_vocabs[(src, i)])
    sub_vocabs[tgt] = Counter()
    bpe_file = bpe_files[(tgt, ac.TRAIN, 100)]
    count_toks(bpe_file, sub_vocabs[tgt])
        
    start_vocab = ac._START_VOCAB
    sorted_keys = joint_vocab.most_common()
    sorted_keys = [kv[0] for kv in sorted_keys]
    vocab_keys = start_vocab + sorted_keys

    with open(joint_vocab_file, 'w') as fout:
        for idx, key in enumerate(vocab_keys):
            count = joint_vocab[key]
            fout.write(f'{key} {idx} {count}\n')

    joint_vocab = {}
    for idx, key in enumerate(vocab_keys):
        joint_vocab[key] = idx

    # get logit mask for each language
    langs = [(src, i) for i in range(0,101,inc)] + [tgt]
    for lang in langs:
        # 0 means masked out, 1 means kept
        mask = np.zeros(len(joint_vocab), dtype=np.uint8)
        mask[ac.UNK_ID] = 1
        mask[ac.EOS_ID] = 1
        for key in sub_vocabs[lang]:
            mask[joint_vocab[key]] = 1
        mask_file = mask_files[lang]
        np.save(mask_file, mask, allow_pickle=True)

    # save all training and dev data as npy files
    for i in range(0,101,inc):
        for mode in [ac.TRAIN, ac.DEV]:
            for lang in [src, tgt]:
                data = []
                bpe_file = bpe_files[(lang, mode, i)]
                with open(bpe_file, 'r') as fin:
                    for line in fin:
                        toks = line.strip().split()
                        toks = [joint_vocab.get(tok, ac.UNK_ID) for tok in toks]
                        if lang == src and eos:
                            toks = toks + [ac.EOS_ID]
                        elif lang == tgt:
                            toks = [ac.BOS_ID] + toks
                        data.append(toks)

                data = np.array(data)
                npy_file = npy_files[(lang, mode, i)]
                np.save(npy_file, data, allow_pickle=True)
