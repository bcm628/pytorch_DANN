import torch
from torch.autograd import Variable
import numpy as np

from train import params
from util import utils


import torch.optim as optim



def train(training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch):
    """
    Execute target domain adaptation
    :param training_mode:
    :param feature_extractor:
    :param class_classifier:
    :param domain_classifier:
    :param class_criterion:
    :param domain_criterion:
    :param source_dataloader:
    :param target_dataloader:
    :param optimizer:
    :return:
    """

    # setup models
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    output_all_mosei = []
    label_all_mosei = []


    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):

        if training_mode == 'dann':
            # setup hyperparameters
            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

            # prepare the data
            input1, label1 = sdata
            input1 = input1.float().cuda()
            label1 = label1.float().cuda()
            #print("input1:", input1.shape, "label1:", label1.shape)
            input2, label2 = tdata
            input2 = input2.float().cuda()
            label2 = label2.float()
            #print(input2.shape, label2.shape)
            # size = min((input1.shape[0], input2.shape[0]))
            # input1, label1 = input1[0:size, :, :], label1[0:size]
            # #print(input1.shape, label1.shape)
            # input2, label2 = input2[0:size, :, :], label2[0:size]

            # with torch.autograd.set_detect_anomaly(True):
            #     if params.use_gpu:
            #         input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            #         input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            #     else:
            #         input1, label1 = Variable(input1), Variable(label1)
            #         input2, label2 = Variable(input2), Variable(label2)

            # setup optimizer
            optimizer = utils.optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            # prepare domain labels
            if params.use_gpu:
                # src_labels1 = Variable(torch.zeros((input1.size()[0])).cuda())
                # src_labels2 = Variable(torch.ones((input1.size()[0])).cuda())
                src_labels1 = torch.zeros((input1.size()[0]))
                src_labels2 = torch.ones((input1.size())[0])
                source_labels = torch.stack((src_labels1, src_labels2), dim=1).cuda()

                # tgt_labels1 = Variable(torch.ones((input2.size()[0])).cuda())
                # tgt_labels2 = Variable(torch.zeros((input2.size()[0])).cuda())
                tgt_labels1 = torch.ones((input2.size()[0]))
                tgt_labels2 = torch.zeros((input2.size()[0]))
                target_labels = torch.stack((tgt_labels1, tgt_labels2), dim=1).cuda()
            else:
                source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
                target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

            # compute the output of source domain and target domain
            src_feature = feature_extractor(input1)
            #print(src_feature.shape) #(20, 74)
            tgt_feature = feature_extractor(input2)
            #print(tgt_feature.shape)

            # compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            #class_preds = class_preds.view(-1,3,2)
            #print("class preds:", class_preds.shape) #(20, 3)
            #class_preds = class_preds.view(-1, 2)
            #label1 = label1.squeeze(dim=2)
            # label1 = label1.view(20,6) #(20, 3, 1)
            class_loss = class_criterion(class_preds, label1)

            # compute the domain loss of src_feature and target_feature
            tgt_preds = domain_classifier(tgt_feature, constant=constant)
            src_preds = domain_classifier(src_feature, constant=constant)
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss


            loss = class_loss + params.theta * domain_loss
            loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), loss.item(), class_loss.item(),
                    domain_loss.item()
                ))


        elif training_mode == 'source':
            # prepare the data
            input1, label1 = sdata
            label1 = label1.long()
            #size = input1.shape[0]
            #input1, label1 = input1[0:size, :, :, :], label1[0:size]

            if params.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)

            # setup optimizer
            optimizer = optim.SGD(list(feature_extractor.parameters())+list(class_classifier.parameters()), lr=0.01, momentum=0.9)
            optimizer.zero_grad()

            # compute the output of source domain and target domain
            src_feature = feature_extractor(input1, embedding_dim=params.mod_dim)

            # compute the class loss of src_feature
            class_preds = class_classifier(src_feature)
            class_preds = class_preds.view(-1, 2)
            label1 = label1.view(-1)
            class_loss = class_criterion(class_preds, label1)

            class_loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input1), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), class_loss.item()
                ))

        elif training_mode == 'target':
            # prepare the data
            input2, label2 = tdata
            size = input2.shape[0]
            input2, label2 = input2[0:size, :, :, :], label2[0:size]
            if params.use_gpu:
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input2, label2 = Variable(input2), Variable(label2)

            # setup optimizer
            optimizer = optim.SGD(list(feature_extractor.parameters()) + list(class_classifier.parameters()), lr=0.01,
                                  momentum=0.9)
            optimizer.zero_grad()

            # compute the output of source domain and target domain
            tgt_feature = feature_extractor(input2)

            # compute the class loss of src_feature
            class_preds = class_classifier(tgt_feature)
            class_loss = class_criterion(class_preds, label2)

            class_loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), class_loss.item()
                ))