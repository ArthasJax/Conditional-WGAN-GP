#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 15:07
# @Author  : Jiazheng Li
# @File    : data_loader.py


import csv

import numpy as np
import pandas as pd
import torch
from numpy import shape
from torch.utils import data

BATCH_SIZE = 256


def load_wind():
    rows = pd.read_csv('./datasets/wind_32.csv', header=None)
    rows = np.array(rows, dtype=float)  # rows是list型，转换成ndarray数组
    rows = rows[:104832, :]
    print(shape(rows))

    m = np.max(rows)
    print("Maximum value of wind", m)
    half_rows = m / 2
    rows = (rows - half_rows) / half_rows

    trX = np.reshape(rows.T, (-1, 576))
    trX = trX.reshape(-1, 1, 24, 24)
    print("Shape TrX", shape(trX))
    trX = torch.tensor(trX, dtype=torch.float32)

    labels = pd.read_csv('./datasets/wind_label.csv', header=None)
    labels = np.array(labels, dtype=int)
    trY = np.tile(labels, (32, 1)).reshape(-1, )
    trY = torch.tensor(trY, dtype=torch.int64)

    train_ids = data.TensorDataset(trX, trY)

    dataloader = data.DataLoader(train_ids, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    return dataloader


# solar.csv=105120*32 solar_label.csv=182*1
# 最大光伏8.13
def load_solar_data():
    with open('datasets/solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)
    print(shape(labels))

    rows = pd.read_csv('./datasets/solar.csv', header=None)
    rows = np.array(rows, dtype=float)
    rows = rows[:104832, :]
    m = np.max(rows)
    print("maximum value of solar power", m)
    half_rows = m / 2
    rows = (rows - half_rows) / half_rows

    print(shape(rows))
    trX = np.reshape(rows.T, (-1, 576))
    trX = trX.reshape(-1, 1, 24, 24)
    trX = torch.tensor(trX, dtype=torch.float32)

    trY = np.tile(labels, (32, 1)).reshape(-1, )  # 将label沿着axis=0轴，复制32遍，axis=1轴不变
    trY = torch.tensor(trY, dtype=torch.int64)

    train_ids = data.TensorDataset(trX, trY)
    dataloader = data.DataLoader(train_ids, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    return dataloader
    # label.shape=(182,1)   trY.shape=(5824,1)   trX.shape=(5824,576)


# def two_year():
#     imgsize = 24
#
#     rows = pd.read_csv('./datasets/two_year.csv')
#     b = pd.to_datetime(rows['DateTime'], format='%d/%m/%Y %H:%M')
#     rows = rows['Measured & upscaled [MW]']
#
#     c = b.dt.month
#     d = np.array(c[:int(70176 / (imgsize ** 2)) * imgsize ** 2]).reshape(-1, imgsize * imgsize)
#     labels = d[:, 0] - 1
#
#     rows = np.array(rows)
#
#     if np.isnan(np.sum(rows)):
#         rows[np.isnan(rows)] = np.mean(rows[~np.isnan(rows)])  # 将nan替换为均值
#
#     trY = torch.tensor(labels, dtype=torch.int64)
#
#     m = np.max(rows)
#     print("maximum value of solar power", m)
#     half_rows = m / 2
#     rows = (rows - half_rows) / half_rows
#
#     trX = np.reshape(rows[:int(70176 / (imgsize ** 2)) * imgsize ** 2], (-1, imgsize*imgsize))
#     trX = trX.reshape((-1, 1, imgsize, imgsize))
#     trX = torch.tensor(trX, dtype=torch.float32)
#
#     train_ids = data.TensorDataset(trX, trY)
#     dataloader = data.DataLoader(train_ids, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#
#     return dataloader


if __name__ == '__main__':
    dataloader = load_solar_data()
    a, b = next(iter(dataloader))
    print(a.shape)
    print(b.shape)
