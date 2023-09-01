import argparse
import os
import json
import pickle as pickle
from collections import defaultdict, Counter
from os.path import dirname, join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train_ci, train_pretrain_model
import utils

from vqa_debias_loss_functions import *

import yaml
from pprint import pprint
import copy


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', action="store_true",
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--nocp', action="store_true", help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--mode', default="b_learned_mixin",
        choices=["b_learned_mixin", "vb_learned_mixin", "learned_mixin", "reweight", "bias_product", "none"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--use_cuda', action="store_true", default=True,
        help="use GPU to run the program")
    parser.add_argument(
        '--eval_each_epoch', action="store_true",
        help="Evaluate every epoch, instead of at the end")

    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pprint(args)

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    # load model option
    if '3.12' in yaml.__version__:
        with open('option/vqa_ci.yaml') as f:
            opt = yaml.load(f)
    else:
        with open('option/vqa_ci.yaml') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

    general_opt = opt['General']
    cp = general_opt['cp']
    output = general_opt['output']
    debug = general_opt['debug']

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, cp=cp,
                                   cache_image_features=args.cache_features)
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, cp=cp,
                                  cache_image_features=args.cache_features)

    answer_voc_size = train_dset.num_ans_candidates

    # Compute the bias:
    # The bias here is just the expected score for each answer/question type

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

    # Record the bias function we are using
    utils.create_dir(output)

    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # The original version uses multiple workers, but that just seems slower on my setup
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=8)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=8)

    args.eval_each_epoch = True

    print("Starting training...")

    # print('training q bias model...')
    model_qonly = base_model.build_model_bceloss(train_dset, args.num_hid)
    # model_qonly.w_emb.init_embedding('data/glove6b_init_300d.npy')
    # train_bceloss(model_qonly, train_loader, eval_loader, args.epochs,
    #               output, args.eval_each_epoch, train_dset.label2ans, debug=debug, train_qbias=True, cp=cp)
    # print('q bias model trained!')

    # print('training base model')
    model_bmup = base_model.build_model_bceloss(train_dset, args.num_hid)

    use_cuda = torch.cuda.is_available() and args.use_cuda
    if use_cuda:
        model_qonly = model_qonly.cuda()
        model_bmup = model_bmup.cuda()
    # model_bmup = base_model.build_model_bcelossci(train_dset, args.num_hid).cuda()
    # model_bmup.w_emb.init_embedding('data/glove6b_init_300d.npy')
    # train_bceloss(model_bmup, train_loader, eval_loader, args.epochs,
    #               output, args.eval_each_epoch, train_dset.label2ans, debug=debug, cp=cp)
    # print('base model trained!')

    # if os.path.exists(f'{output}/lhm_model_epoch14_qonly.pth'):
    if os.path.exists(f'{output}/model_epoch14_qonly.pth'):
        model_qonly_state_dict = torch.load(f'{output}/lhm_model_epoch14_qonly.pth')
        model_qonly.load_state_dict(model_qonly_state_dict)
    else:
        print('training q bias model...')
        model_qonly.w_emb.init_embedding('data/glove6b_init_300d.npy')
        train_pretrain_model(model_qonly, train_loader, eval_loader, args.epochs,
                             output, train_dset.label2ans, use_cuda=use_cuda, debug=debug, train_qbias=True, cp=cp)
        print('q bias model trained!')

    # if os.path.exists(f'{output}/lhm_model_epoch14_bmup.pth'):
    if os.path.exists(f'{output}/model_epoch14_bmup.pth'):
        model_bmup_state_dict = torch.load(f'{output}/lhm_model_epoch14_bmup.pth')
        model_bmup.load_state_dict(model_bmup_state_dict)
    else:
        print('training base model')
        model_bmup.w_emb.init_embedding('data/glove6b_init_300d.npy')
        train_pretrain_model(model_bmup, train_loader, eval_loader, args.epochs,
                             output, train_dset.label2ans, use_cuda=use_cuda, debug=debug, cp=cp)
        print('base model trained!')

    student_model = base_model.build_model_bceloss(train_dset, args.num_hid, use_sigmoid=False).cuda()
    student_model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model_ci = base_model.CI_Model(args.alpha, student_model).cuda()

    train_ci(model_bmup, model_qonly, model_ci, train_loader, eval_loader, args.epochs,
                output, train_dset.label2ans, debug=debug, cp=cp)


if __name__ == '__main__':
    main()
