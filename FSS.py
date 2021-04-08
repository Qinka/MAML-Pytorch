import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import random
import glob


class FSSDataset(Dataset):

    def __init__(self, root, mode, batchsz, k_shot, k_query, resize, start_idx = 0):
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = 2  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.start_idx = start_idx  # index label not from 0, but from startidx
        print(f"shuffle DB :{mode}, b:{batchsz} , 2-way, {k_shot}-shot, {k_query}-query, resize:{resize}")

        if mode == 'train':
            self.transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((self.resize, self.resize)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
        else:
            self.transform = transforms.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        self.label = transforms.Compose([
                lambda x: Image.open(x).convert('L'),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                lambda x: x.float(),
            ])

        self.path = root
        self.data = {}

        files = glob.glob(os.path.join(self.path, '*', '*.jpg'))

        for f in files:
            t, i = f.split(os.path.sep)[-2:]
            i = i.replace('.jpg', '')

            if t not in self.data:
                self.data[t] = []
            p = lambda t, i, s: os.path.join(self.path, t, i + s)
            self.data[t].append((p(t, i, '.jpg'), p(t, i, '.png')))

        self.cls_num = len(list(self.data.keys()))

        self.create_batch(self.batchsz)


    def create_batch(self, batchsz):
        self.support_x_batch = []
        self.query_x_batch   = []
        for b in range(batchsz):
            support_x = []
            query_x   = []
            cls = np.random.choice(list(self.data.keys()), 1, False)[0]
            selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)

            indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
            indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
            support_x.append(
                np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
            query_x.append(np.array(self.data[cls])[indexDtest].tolist())
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)


    def __getitem__(self, index):
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = torch.FloatTensor(self.setsz, 1, self.resize, self.resize)
        # [querysz, 3, resize, resize]
        query_x   = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y   = torch.FloatTensor(self.querysz, 1, self.resize, self.resize)

        flatten_support = \
            [item for sublist in self.support_x_batch[index] for item in sublist]

        flatten_query = \
            [item for sublist in self.query_x_batch[index] for item in sublist]

        for i, path in enumerate(flatten_support):
            support_x[i] = self.transform(path[0])
            support_y[i] = self.label(path[1])

        for i, path in enumerate(flatten_query):
            query_x[i] = self.transform(path[0])
            query_y[i] = self.label(path[1])

        return support_x, support_y, query_x, query_y

    def __len__(self):
        return self.batchsz

