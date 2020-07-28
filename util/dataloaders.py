import torch
from torch.utils.data import DataLoader
from data.iemocap import MultimodalDataset
from data.mosei import MoseiNewDataset
from train import params

#TODO: make batch size so that the number of samples is divisible by it

###attempt 1: based on FMT code###

def get_train_loader(dataset):
    if dataset == 'iemocap':
        ds = MultimodalDataset
        path = params.iemocap_path

    elif dataset == 'mosei':
        ds = MoseiNewDataset
        path = params.mosei_path

    train_dataset = ds(path, mod=params.modality, cls="train")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=params.batch_size,
                              shuffle=True,
                              num_workers=1,
                              drop_last=True,)

    return train_loader


def get_test_loader(dataset):
    if dataset == 'iemocap':
        ds = MultimodalDataset
        path = params.iemocap_path

    elif dataset == 'mosei':
        ds = MoseiNewDataset
        path = params.mosei_path

    test_dataset = ds(path, mod=params.modality, cls="test")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=params.batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=True,)

    return test_loader


def get_valid_loader(dataset):
    if dataset == 'iemocap':
        ds = MultimodalDataset
        path = params.iemocap_path

    elif dataset == 'mosei':
        ds = MoseiNewDataset
        path = params.mosei_path

    valid_dataset = ds(path, mod=params.modality, cls="valid")
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=params.batch_size,
                              shuffle=False,
                              num_workers=1,
                              drop_last=True,)

    return valid_loader
