import torch
from torchvision import transforms
import torch.utils.data as Data
import os
import io
import numpy as np
import cv2
from glob import glob


class CustomDataset(Data.Dataset):
    def __len__(self):
        return len(self._img_frames)

    def __init__(self, data_path="./data/membrane/train", img_file="image", mask_file="label", transform=None):
        super(CustomDataset, self).__init__()
        self._img_frames = sorted(glob(os.path.join(data_path, img_file, "*")))
        self._mask_frames = sorted(glob(os.path.join(data_path, mask_file, "*")))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self._img_frames[index]
        mask_path = self._mask_frames[index]
        img = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        # mask = cv2.resize(mask, (256, 256))
        img = img.astype('float32') / 255.0
        mask = mask.astype('float32') / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        img = np.array(img).reshape((1, img.shape[0], img.shape[1]))
        mask = np.array(mask).reshape((1, mask.shape[0], mask.shape[1]))
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        return (img, mask)


class DataLoader(object):
    def __init__(self, train_path="./data/membrane/train", test_path="./data/membrane/test", img_file="image", mask_file="label", transform=None, img_size=(512, 512)):
        self._train_dataset = CustomDataset(train_path, img_file, mask_file, transform)
        self._test_path = test_path
        self._img_size = img_size

    def load_test_data(self):
        imgs = sorted(glob(os.path.join(self._test_path, "*")))
        x_test = []
        for data in imgs:
            img = cv2.imread(data, 0)
            img = cv2.resize(img, self._img_size)
            img = img.astype('float32') / 255.0
            x_test.append(img)
        x_test = np.array(x_test).reshape((len(x_test), 1, self._img_size[0], self._img_size[1]))
        x_test = torch.from_numpy(x_test)
        return x_test

    def load_train_data(self, shuffle=True, batch_size=2, num_workers=4):
        return Data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle)
