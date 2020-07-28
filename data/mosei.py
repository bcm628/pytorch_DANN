import os
import pickle

import numpy as np
import torch
import torch.utils.data as Data

from data.iemocap import MultimodalSubdata
#from consts import global_consts as gc
from train import params
from util.process_MOSEI import format_mosei

class MoseiNewDataset(Data.Dataset):
    #TODO: figure out what is happening here
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train", mod='acoustic'):
        self.root = root
        self.cls = cls
        self.mod = mod
        # if len(MoseiNewDataset.trainset.y) != 0 and cls != "train":
        #     print("Data has been previously loaded, fetching from previous lists.")
        # else:
        self.load_data(mod)

        if self.cls == "train":
            self.dataset = MoseiNewDataset.trainset
        elif self.cls == "test":
            self.dataset = MoseiNewDataset.testset
        elif self.cls == "valid":
            self.dataset = MoseiNewDataset.validset

#TODO: make sure i fixed this
        self.feat = self.dataset.feat
        #self.acoustic = self.dataset.audio
        #self.visual = self.dataset.vision
        self.y = self.dataset.y


    def load_data(self, modality):
        dataset = format_mosei(os.path.join(params.mosei_path, 'tensors.pkl'), three_dim=True)

        # if modality == 'text':
        #     #params.padding_len = dataset['test']['language'].shape[1]
        #     params.mod_dim = dataset['test']['language'].shape[2]

        if modality == 'acoustic':
            #params.padding_len = dataset['test']['language'].shape[1]
            params.mod_dim = dataset['test']['acoustic'].shape[2]

        elif modality == 'visual':
            #params.padding_len = dataset['test']['language'].shape[1]
            params.mod_dim = dataset['test']['visual'].shape[2]

        for ds, split_type in [(MoseiNewDataset.trainset, 'train'), (MoseiNewDataset.validset, 'valid'),
                               (MoseiNewDataset.testset, 'test')]:

            # if modality == 'text':
            #     ds.feat = torch.tensor(dataset[split_type]['language'].astype(np.float32)).cpu().detach()

            if modality == 'acoustic':
                ds.feat = torch.tensor(dataset[split_type]['acoustic'].astype(np.float32))
                ds.feat[ds.feat == -np.inf] = 0
                ds.feat = ds.feat.clone().cpu().detach()

            elif modality == 'visual':
                ds.feat = torch.tensor(dataset[split_type]['visual'].astype(np.float32)).cpu().detach()

            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()


    def __getitem__(self, index):
        inputLen = len(self.feat[index])
        return self.feat[index], self.y[index].squeeze()
        # inputLen = len(self.language[index])
        # return self.language[index], self.acoustic[index], self.visual[index], \
        #        inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MoseiNewDataset(params.mosei_path)
