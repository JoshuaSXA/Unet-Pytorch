import torch
from torchvision import transforms
import torch.utils.data as Data
import os
import numpy as np
from glob import glob
from PIL import Image
from data_aug import *


class CustomDataset(Data.Dataset):
    def __len__(self):
        return len(self._img_frames)

    def __init__(self, image_frames, mask_frames, transform=None):
        super(CustomDataset, self).__init__()
        self._img_frames = image_frames
        self._mask_frames = mask_frames
        self._transform = transform

    def __getitem__(self, index):
        img_path = self._img_frames[index]
        mask_path = self._mask_frames[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self._transform is not None:
            img, mask = self._transform(img, mask)
        img, mask = image_resize(img, mask, (512, 512))

        img = np.array(img).astype("float32")
        mask = np.array(mask).astype("float32")
        img, mask = image_to_tensor(img, mask)
        return (img, mask)


class ISICDataLoader(object):
    def __init__(self, image_path, mask_path, split_ratio=0.05,transforms=None):
        img_frames = sorted(glob(os.path.join(image_path, "*")))
        mask_frames = sorted(glob(os.path.join(mask_path, "*")))
        self._split_ratio = split_ratio
        train_frames, val_frames = self.split_dataset(img_frames, mask_frames)
        self._train_dataset = CustomDataset(train_frames['image'], train_frames['mask'], transforms)
        self._val_dataset = CustomDataset(val_frames['image'], val_frames['mask'], transforms)

    def split_dataset(self, img_frames, mask_frames):
        total_len = len(img_frames)
        val_len = round(total_len * self._split_ratio)
        train_img_frames = img_frames[:-val_len]
        train_mask_frames = mask_frames[:-val_len]
        val_img_frames = img_frames[-val_len:]
        val_mask_frames = mask_frames[-val_len:]
        return {"image":train_img_frames, "mask":train_mask_frames}, {"image":val_img_frames, "mask":val_mask_frames}

    def get_train_dataloader(self, batch_size=4, shuffle=True, num_works=4):
        return Data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_works)

    def get_val_dataloader(self, batch_size=4, num_works=4):
        return Data.DataLoader(self._val_dataset, batch_size=batch_size, num_workers=num_works)



# dataloader = ISICDataLoader(image_path="./data/ISIC2018/image/", mask_path="./data/ISIC2018/mask/")
# val_loader = dataloader.get_val_dataloader(num_works=0)
# for (img, mask) in val_loader:
#     print(img.shape, mask.shape)
#     break