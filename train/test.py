"""
Test the model with target domain
"""
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from train import params
from models import models


def eval_data(label_all, output_all, set):
    truths = np.array(label_all)
    results = np.array(output_all)
    test_preds = results.reshape((-1, 3, 2))
    print(np.shape(test_preds))
    test_truth = truths.reshape((-1, 3))
    f1_total = {}
    acc_total = {}

    for i, emo in enumerate(params.emo_labels):
        test_preds_i = np.argmax(test_preds[:, i], axis=1)
        test_truth_i = test_truth[:, i]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        f1_total[emo] = f1
        acc_total[emo] = acc
        print("\t%s %s F1 score: %f" % (set, params.emo_labels[i], f1))
        print("\t%s %s Accuracy score: %f" % (set, params.emo_labels[i], acc))
    return f1_total, acc_total


def test(feature_extractor, class_classifier, domain_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0
    domain_correct = 0.0
    tgt_correct = 0.0
    src_correct = 0.0

    src_output_all = []
    src_label_all = []
    tgt_output_all = []
    tgt_label_all = []


    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        input1, label1 = sdata
        if params.use_gpu:
            input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
        else:
            input1, label1 = Variable(input1), Variable(label1)
            src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))

        output1 = class_classifier(feature_extractor(input1,
                                                     embedding_dim=params.mod_dim,
                                                     num_layers=params.extractor_layers))

        #print("output1:", output1.shape)
        output1 = output1.view(-1, 2)
        #print("output1:", output1.shape)
        #print("label1:", label1.shape)
        label1 = label1.view(-1)
        #print("label1:", label1.shape)

        src_output_all.extend(output1.tolist())
        src_label_all.extend(label1.tolist())

        #pred1 = output1.data.max(1, keepdim = True)[1]
        # print(pred1.shape)
        # source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()
        #
        # src_preds = domain_classifier(feature_extractor(input1), constant)
        # src_preds = src_preds.data.max(1, keepdim=True)[1]
        # src_correct += src_preds.eq(src_labels.data.view_as(src_preds)).cpu().sum()

    print(len(src_output_all), len(src_label_all))


    src_f1, src_acc = eval_data(src_label_all, src_output_all, "source")
    quit()


    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2 = tdata
        #print(input2.shape, label2.shape)
        if params.use_gpu:
            input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
        else:
            input2, label2 = Variable(input2), Variable(label2)
            tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

        output2 = class_classifier(feature_extractor(input2,
                                                     embedding_dim=params.mod_dim,
                                                     num_layers=params.extractor_layers))


        #print("output2:", output2.shape)
        #TODO: output2 should be [20,8]
        output2 = output2.view(-1, 2)
        #print("output2:", output2.shape)
        #print("label2:", label2.shape)
        label2 = label2.view(-1)
        #print("label2:", label2.shape)

        tgt_output_all.extend(output2.tolist())
        #print(len(tgt_output_all))
        tgt_label_all.extend(label2.tolist())
        #print(len(tgt_label_all))

    print(len(tgt_output_all), len(tgt_label_all))
    quit()
    tgt_f1, tg_acc = eval_data(tgt_label_all, tgt_output_all, "target")

        # pred2 = output2.data.max(1, keepdim=True)[1]
        # target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()
        #
        # tgt_preds = domain_classifier(feature_extractor(input2,
        #                                              embedding_dim=params.mod_dim,
        #                                              num_layers=params.extractor_layers),
        #                               constant)
        # tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        # tgt_correct += tgt_preds.eq(tgt_labels.data.view_as(tgt_preds)).cpu().sum()


    #domain_correct = tgt_correct + src_correct



    # print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
    #       'Domain Accuracy: {}/{} ({:.4f}%)\n'.
    #     format(
    #     source_correct, len(source_dataloader.dataset), 100. * float(source_correct) / len(source_dataloader.dataset),
    #     target_correct, len(target_dataloader.dataset), 100. * float(target_correct) / len(target_dataloader.dataset),
    #     domain_correct, len(source_dataloader.dataset) + len(target_dataloader.dataset),
    #     100. * float(domain_correct) / (len(source_dataloader.dataset) + len(target_dataloader.dataset))
    # ))