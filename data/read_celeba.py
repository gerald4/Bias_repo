"""
This code is taken from https://github.com/grayhong/bias-contrastive-learning/blob/master/debias/datasets/celeba.py
"""



import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA




class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by


def get_unsup_confusion_matrix(num_classes, targets, biases, marginals):
    confusion_matrix_org = torch.zeros(num_classes, num_classes).float()
    confusion_matrix_cnt = torch.zeros(num_classes, num_classes).float()
    for t, p, m in zip(targets, biases, marginals):
        confusion_matrix_org[p.long(), t.long()] += m
        confusion_matrix_cnt[p.long(), t.long()] += 1

    zero_idx = confusion_matrix_org == 0
    confusion_matrix_cnt[confusion_matrix_cnt == 0] = 1
    confusion_matrix_org = confusion_matrix_org / confusion_matrix_cnt
    confusion_matrix_org[zero_idx] = 1
    confusion_matrix_org = 1 / confusion_matrix_org
    confusion_matrix_org[zero_idx] = 0

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix


class BiasedCelebASplit:
    def __init__(self, root, split, transform, target_attr = "makeup", with_ind = True, with_bias_att = True, **kwargs):
        self.transform = transform
        self.target_attr = target_attr

        #gnanfack edit, transforming test to valid and valid to train_valid
        if split == "test":
            split = "valid"
        if split == "valid":
            split = "train_valid"

        
        self.celeba = CelebA(
            root=root,
            split="train" if split == "train_valid" else split,
            target_type="attr",
            transform=transform
        )
        self.bias_idx = 20
        
        if target_attr == 'blonde':
            self.target_idx = 9
            if split in ['train', 'train_valid']:
                save_path = Path(root) / 'pickles' / 'blonde'
                if save_path.is_dir():
                    print(f'use existing blonde indices from {save_path}')
                    self.indices = pickle.load(open(save_path / 'indices.pkl', 'rb'))
                else:
                    self.indices = self.build_blonde()
                    print(f'save blonde indices to {save_path}')
                    save_path.mkdir(parents=True, exist_ok=True)
                    pickle.dump(self.indices, open(save_path / f'indices.pkl', 'wb'))
                self.attr = self.celeba.attr[self.indices]
            else:
                self.attr = self.celeba.attr
                self.indices = torch.arange(len(self.celeba))

        elif target_attr == 'makeup':
            self.target_idx = 18
            self.attr = self.celeba.attr
            self.indices = torch.arange(len(self.celeba))
        else:
            raise AttributeError
            
        if split in ['train', 'train_valid']:
            save_path = Path(f'clusters/celeba_rand_indices_{target_attr}.pkl')
            if not save_path.exists():
                rand_indices = torch.randperm(len(self.indices))
                pickle.dump(rand_indices, open(save_path, 'wb'))
            else:
                rand_indices = pickle.load(open(save_path, 'rb'))
            
            num_total = len(rand_indices)
            num_train = int(0.8 * num_total)
            
            if split == 'train':
                indices = rand_indices[:num_train]
            elif split == 'train_valid':
                indices = rand_indices[num_train:]
            
            self.indices = self.indices[indices]
            self.attr = self.attr[indices]

        self.targets = self.attr[:, self.target_idx]
        self.biases = self.attr[:, self.bias_idx]

        self.confusion_matrix_org, self.confusion_matrix, self.confusion_matrix_by = get_confusion_matrix(num_classes=2,
                                                                                                          targets=self.targets,
                                                                                                          biases=self.biases)


        #gnanfack edit // Adding the possibility to not return biased attributes and index
        self.with_ind = with_ind
        self.with_bias_att = with_bias_att


        print(f'Use BiasedCelebASplit \n target_attr: {target_attr} split: {split} \n {self.confusion_matrix_org}')

    def build_blonde(self):
        biases = self.celeba.attr[:, self.bias_idx]
        targets = self.celeba.attr[:, self.target_idx]
        selects = torch.arange(len(self.celeba))[(biases == 0) & (targets == 0)]
        non_selects = torch.arange(len(self.celeba))[~((biases == 0) & (targets == 0))]
        np.random.shuffle(selects)
        indices = torch.cat([selects[:2000], non_selects])
        return indices

    def __getitem__(self, index):
        img, _ = self.celeba.__getitem__(self.indices[index])
        target, bias = self.targets[index], self.biases[index]

        if not(self.with_ind) and not(self.with_bias_att):        
            return img, target

        return img, torch.tensor([target, bias]), index

    def __len__(self):
        return len(self.targets)


