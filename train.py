#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 15:07
# @Author  : Jiazheng Li
# @File    : train.py

import os

import torch
import torch.optim as optim
import torchvision
from utils import gradient_penalty, save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt
from data_loader import load_solar_data, load_wind
from model import Discriminator, Generator, initialize_weights

DATASET = 'solar'  # 'solar' 'wind'

if DATASET == 'solar':
    loader = load_solar_data()
elif DATASET == 'wind':
    loader = load_wind()

# 超参数设置，可能需要修改IMAGE_SIZE
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4  # 改成5e-4训练会快点
CHANNELS_IMG = 1
IMAGE_SIZE = 24
NUM_CLASSES = 12
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 300
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

if not os.path.exists('images'):
    os.makedirs('images')


# 初始化生成器和判别器  note: discriminator 就是 critic,
# 因为判别器不再输出 [0, 1] 之间的数
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# 初始化优化器
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # 训练判别器: max E[critic(real)] - E[critic(fake)]
        # 相当于最小化上式
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # 训练生成器: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                fake = fake.detach().cpu()
                test_images = fake.reshape(-1, IMAGE_SIZE * IMAGE_SIZE)

            fig, axs = plt.subplots(5, 5, dpi=100)
            cnt = 0
            for i in range(5):
                for j in range(5):
                    axs[i, j].plot(test_images[cnt, :288])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/%d.png" % epoch)
            # plt.show()
            plt.close()

            step += 1


## 保存模型
if not os.path.exists('model'):
    os.makedirs('model')

torch.save(gen, 'model/' + 'generator_WGAN.pkl')
torch.save(critic, 'model/' + 'discriminator_WGAN.pkl')
