import os
import torch.optim as optim
import torch.nn as nn
from model.unet import UNet
from load import *
import utils

# choose the device gpu / cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = UNet()
net = net.to(device)
net.apply(utils.weights_initializer)


optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.BCELoss()

dataloader = DataLoader(transform=None)
train_loader = dataloader.load_train_data()
test_data = dataloader.load_test_data().to(device)

def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print("Batch %d: Train Loss is %.5f" % (batch_idx, loss))


def test():
    net.eval()
    with torch.no_grad():
        img = test_data[0]
        dim = list(img.size())
        img = img.view(1, dim[0], dim[1], dim[2])
        out = net(img).to("cpu")
        utils.show_image(out)





for i in range(30):
    train(i + 1)
torch.save(net, './model.pkl')
test()