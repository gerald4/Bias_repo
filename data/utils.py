''' This code is taken from 
https://github.com/kakaoenterprise/Learning-Debiased-Disentangled/blob/master/data/util.py
'''


import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image



from functools import partial
from .read_celeba import BiasedCelebASplit


class CMNISTDataset(Dataset):
    def __init__(self, root, split, transform=None, with_ind = True, with_bias_att = True):
        super(CMNISTDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.with_ind = with_ind
        self.with_bias_att = with_bias_att
        if split=='train':
            self.align = glob(os.path.join(self.root, 'align', '*', '*'))
            self.conflict = glob(os.path.join(self.root, 'conflict', '*', '*'))
            self.data = self.align + self.conflict
        elif split=='valid':
            self.data = glob(os.path.join(self.root,split,"*"))
        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))
        else: "Not 'train', 'valid', or 'test' !"

        self.labels = [int(self.data[ind].split("\\")[-1].split('_')[-2]) for ind in range(len(self.data))]

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        # read image
        img = Image.open(self.data[idx]).convert('RGB')
        # print('idx', idx)
        # print(self.data[idx])
        # print(self.data[idx].split("\\")[-1].split('_')[-2])
        # print(self.data[idx].split("\\")[-1].split('_')[-1].split('.')[0])

        # transforms image
        if self.transform is not None:
            img = self.transform(img)
        # Get (label, color label)
        labels = torch.LongTensor([int(self.data[idx].split("\\")[-1].split('_')[-2]), int(self.data[idx].split("\\")[-1].split('_')[-1].split('.')[0])])
        # return image, (label, color label), name of img
        if not(self.with_ind) and not(self.with_bias_att):
            labels = labels[0]
            return img, labels

        return img, labels, self.data[idx]


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split=='train':
            self.align = glob(os.path.join(root, 'align',"*","*"))
            self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
            self.data = self.align + self.conflict

        elif split=='valid':
            self.data = glob(os.path.join(root,split,"*", "*"))

        elif split=='test':
            self.data = glob(os.path.join(root, '../test',"*","*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor(
            [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr, self.data[index]


        
    
transforms = {
    "cmnist": {
        "train": T.Compose([T.ToTensor()]),
        "eval": T.Compose([T.ToTensor()]),
        "test": T.Compose([T.ToTensor()]),
        },
    "cifar10c": {
        "train_aug": T.Compose(
            [   
                T.ToTensor(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "train": T.Compose(
            [
                T.ToTensor(),
                T.RandomResizedCrop(32),
                T.RandomHorizontalFlip(),
                # T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
          "test": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },

    "celeba": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
}


classes = {

        "cmnist": 10,
        "celeba": 2,
}


def get_dataset(dataset, data_dir, dataset_split, percent = "5pct", use_preprocess = None, 
    image_path_list = None, use_type0=None, use_type1=None, with_ind = True, with_bias_att = True, task = "makeup"):
    

    dataset_category = "eval" if (dataset_split == "valid") else dataset_split


    if use_preprocess:
        transform = transforms_preprcs[dataset_category][dataset_split]
    else:
        transform = transforms[dataset][dataset_category]


    if dataset == 'cmnist':
        root = data_dir + f"/cmnist/{percent}"
        dataset = CMNISTDataset(root = root, split = dataset_split,transform=transform, with_ind = with_ind, with_bias_att = with_bias_att)

    elif dataset == "celeba":
        root = f"{data_dir}/celeba/"
        dataset = BiasedCelebASplit(
        root = root,
        split = dataset_split,
        transform = transform,
        target_attr = task,
        with_ind = with_ind,
        with_bias_att = with_bias_att
    )

    return dataset



def get_number_classes(dataset):

    return classes[dataset]