import torch.optim as optim
import torch.nn as nn
from model.unet import UNet
from load import *
from dice_loss import *
import utils


def eval_net(net, dataloader):
    """Evaluation without the densecrf with the dice coefficient"""

    # choose the device gpu / cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net
    net = net.to(device)
    net.eval()
    tot = 0
    for (img, mask) in dataloader:
        inputs, targets = img.to(device), mask.to(device)
        outputs = net(inputs)
        len = list(outputs.size())[0]

        for i in range(len):
            mask_pred = outputs[i]
            true_mask = targets[i]
            mask_pred = (mask_pred > 0.5).float()
            tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)



def train_net(net, epochs=10, batch_size=4,lr=0.1, val_percent=0.05, save_cp=True):
    # choose the device gpu / cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # net
    net = net.to(device)
    net.apply(utils.weights_initializer)

    # dataloader
    dataloader = ISICDataLoader(image_path="./data/ISIC2018/image/", mask_path="./data/ISIC2018/mask/")
    train_loader = dataloader.get_train_dataloader(batch_size=4, shuffle=True, num_works=0)
    val_loader = dataloader.get_val_dataloader(batch_size=4, num_works=0)

    # optimizer and criterion
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.BCELoss()

    dir_checkpoint = './checkpoints/'

    # train
    for epoch in range(epochs):
        # epoch_train_loss = 0.0
        # net.train()
        # for (img, mask) in train_loader:
        #     inputs, targets = img.to(device), mask.to(device)
        #     optimizer.zero_grad()
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        #     optimizer.step()
        #     epoch_train_loss += loss.item()
        val_dice_loss = eval_net(net, val_loader)
        print(val_dice_loss)
        # print('Epoch %d, total train loss is %.5f, test dice loss is %.5f' % ((epoch + 1),epoch_train_loss, val_dice_loss) )
        #
        # if save_cp:
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
        #     print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    train_net(net)
