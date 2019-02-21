from PIL import Image
import random
import math
import torch
import types

# Random rotation.
def random_rotate(img, mask, degrees):
    if isinstance(degrees, int):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    random_degree = random.randint(degrees[0], degrees[1])
    img_rotate = img.rotate(random_degree)
    mask_rotate = mask.rotate(random_degree)
    return img_rotate, mask_rotate

# Apply random cropping to original images, and resize the cropped result to a fixed size.
def random_resized_crop(img, mask, size=(512, 512), scale=(0.6, 1.0), ratio=(4. / 5., 5. / 4.)):
    area = img.size[0] * img.size[1]
    target_area = random.uniform(*scale) * area
    aspect_ratio = random.uniform(*ratio)
    w = int(round(math.sqrt(target_area * aspect_ratio)))
    h = int(round(math.sqrt(target_area / aspect_ratio)))
    w = min(w, img.size[0])
    h = min(h, img.size[1])
    axis_x = round(random.uniform(0, img.size[0] - w))
    axis_y = round(random.uniform(0, img.size[1] - h))
    box = (axis_x, axis_y, axis_x + w, axis_y + h)
    print(box)
    img_roi = img.crop(box)
    img_roi = img_roi.resize(size)
    mask_roi = mask.crop(box)
    mask_roi = mask_roi.resize(size)
    return img_roi, mask_roi

# Apply random horizontal flip to original images with 0.5 as probability.
def random_horizontal_flip(img, mask):
    if_trans = True if random.random() >= 0.5 else False
    if if_trans:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

# Apply random vertical flip to original images with 0.5 as probability.
def random_vertical_flip(img, mask):
    if_trans = True if random.random() >= 0.5 else False
    if if_trans:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def image_resize(img, mask, size=(512, 512)):
    return img.resize(size), mask.resize(size)

def image_to_tensor(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    mask = torch.from_numpy(mask).unsqueeze(0)
    return img, mask



#
# def compose(image, mask, trans_opt=[]):
#     for i in range(len(trans_opt)):
#         trans_func = trans_opt[i]
#         if not isinstance(trans_func, types.FunctionType):
#             assert "Invalid function type."
#             return image, mask
#         image, mask = trans_func(image, mask)