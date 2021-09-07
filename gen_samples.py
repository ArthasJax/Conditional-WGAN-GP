#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/23 17:10
# @Author  : Jiazheng Li
# @File    : gen_samples.py

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
torch.cuda.empty_cache()

DATASET = 'wind'

if DATASET == 'solar':
    img_size = 24
    img_shape = (1, img_size, img_size)
    MAX_NUM = 8.13
elif DATASET == 'wind':
    img_size = 24
    img_shape = (1, img_size, img_size)
    MAX_NUM = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.load('./results/models/generator_WGAN.pkl')

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

noise = Variable(FloatTensor(np.random.normal(0, 1, (2000, 100, 1, 1))))
labels = Variable(LongTensor(np.random.randint(0, 12, 2000)))

out = generator(noise, labels).view(-1, img_size * img_size)
out = out.cpu().detach().numpy()
sample = out * (MAX_NUM / 2) + (MAX_NUM / 2)  # 改成输入数据最大值的一半

windFrame = pd.DataFrame(sample)
windFrame.to_csv('samples.csv', header=None, index=None)

labels = pd.DataFrame(labels.cpu().numpy())
labels.to_csv('labels.csv', header=None, index=None)

fig, axs = plt.subplots(5, 5, dpi=300)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i, j].plot(sample[cnt, :288])
        axs[i, j].axis('off')
        axs[i, j].set_ylim([0, 8])
        cnt += 1
plt.show()
