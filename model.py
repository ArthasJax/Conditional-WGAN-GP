#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 17:10
# @Author  : Jiazheng Li
# @File    : model.py

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size

        self.net = nn.Sequential(
            self._block(channels_noise + embed_size, features_g * 16, 5, 2, 1),  # img: 3 x 3
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 6 x 6
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 12 x 12
            nn.ConvTranspose2d(features_g * 4, channels_img, kernel_size=4, stride=2, padding=1),  # 24 x 24
            nn.Tanh()
        )

        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z, labels):
        # laent vactor z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, embedding], dim=1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),  # 12 x 12
            nn.LeakyReLU(0.2),

            self._block(features_d, features_d * 2, 4, 2, 1),  # 6 x 6
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 3 x 3
            nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=1),  # 1 x 1
        )
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)  # N X C X img_size(H) X img_size(W)
        return self.disc(x)


def initialize_weights(model):
    # 初始化权重
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == '__main__':
    gen = Generator(100, 1, 16, 10, 64, 100)
    dis = Discriminator(1, 16, 10, 24)
    z = torch.rand((32, 100, 1, 1))
    labels = torch.randint(0, 10, (32,))

    x = torch.rand((32, 1, 24, 24))
    output = dis(x, labels)

    print(output.shape)
