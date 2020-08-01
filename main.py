"""
Main script for models
"""
#TODO: add pickling of output representations
#TODO: modify so class classifier only uses three dimensions

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

import numpy as np
import pickle

from models import models
from train import test, train, params
from util import utils
from util import dataloaders

from sklearn.manifold import TSNE

import argparse, sys, os

import torch
from torch.autograd import Variable

import time



def main(args):

    # Set global parameters.
    #params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    source_domain = args.source_domain
    print("source domain is: ", source_domain)
    target_domain = args.target_domain
    print("target domain is: ", target_domain)

    params.modality = args.modality
    print("modality is :", params.modality)
    params.extractor_layers = args.extractor_layers
    print("number of layers in feature extractor: ", params.extractor_layers)
    #params.class_layers = args.class_layers
    #params.domain_layers  = args.domain_layers
    lr = args.lr

    #set output dims for classifier
    #TODO: change this to len of params dict?
    if source_domain == 'iemocap':
        params.output_dim = 4
    elif source_domain == 'mosei':
        params.output_dim = 6


    # prepare the source data and target data

    src_train_dataloader = dataloaders.get_train_loader(source_domain)
    src_test_dataloader = dataloaders.get_test_loader(source_domain)
    src_valid_dataloader = dataloaders.get_valid_loader(source_domain)
    tgt_train_dataloader = dataloaders.get_train_loader(target_domain)
    tgt_test_dataloader = dataloaders.get_test_loader(target_domain)
    tgt_valid_dataloader = dataloaders.get_valid_loader(target_domain)

    print(params.mod_dim)

    # init models
    #model_index = source_domain + '_' + target_domain

    feature_extractor = models.Extractor()
    class_classifier = models.Class_classifier()
    domain_classifier = models.Domain_classifier()
    # feature_extractor = params.extractor_dict[model_index]
    # class_classifier = params.class_dict[model_index]
    # domain_classifier = params.domain_dict[model_index]

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # init optimizer
    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr=lr)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion,
                    domain_criterion, src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
        test.test(feature_extractor, class_classifier, domain_classifier, src_valid_dataloader, tgt_valid_dataloader,
                  epoch)
        if epoch == params.epochs - 1:
            test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader, epoch, mode='test')
        else:
            continue





def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'mosei', help= 'Choose source domain.')

    parser.add_argument('--target_domain', type= str, default= 'iemocap', help = 'Choose target domain.')

    #parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    #parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')

    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')

    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')

    #parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')

    parser.add_argument('--lr', type= float, default= 0.001, help= 'Learning rate.')

    parser.add_argument('--modality', type=str, default='acoustic', help='specify modality: acoustic or visual.')

    parser.add_argument('--extractor_layers', type = int, default=3, help='number of layers in feature extractor CNN')

    #parser.add_argument('--class_layers', type=int, default=2, help='number of layers in class classifier')

    #parser.add_argument('--domain_layers', type=int, default=2, help='number of layers in domain classifier')

    #parser.add_argument('--classifier_type', type=str, default='DNN', help='type of class classifier: "DNN" or "CNN"')

    return parser.parse_args()



if __name__ == '__main__':
    start_time = time.time()
    print('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
    main(parse_arguments(sys.argv[1:]))
    time_passed = time.time() - start_time
    print('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(time_passed)))
