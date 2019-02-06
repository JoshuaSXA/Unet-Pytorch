import os
import torch.optim as optim
import torch.nn as nn
from model.unet import UNet
from load import *
import utils



net = UNet()
net.apply(utils.weights_initializer)


optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.BCELoss()

dataloader = DataLoader(transform=None)
train_loader = dataloader.load_train_data()


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print("Batch %d: Train Loss is %.5f" % (batch_idx, loss))

        

train(1)