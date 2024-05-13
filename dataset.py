import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
import PIL.Image as Image

MORPH_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

FGNET_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


def get_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    dir_path = Path(dir_path)
    classes = []  # 类别名列表

    for category in dir_path.iterdir(): #当path为文件夹时，通过yield产生path文件夹下的所有文件、文件夹路径的迭代器
        if category.is_dir(): # 判断该路径是否是文件夹
            classes.append(category.name)
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表

    for index, name in enumerate(classes):
        class_path = dir_path / name
        if not class_path.is_dir():
            continue
        for img_path in class_path.glob('*.jpg'):
            images_list.append(str(img_path))
            labels_list.append(int(index))
    return images_list, labels_list

def get_MORPH_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    labels_list = []
    images_list = []
    dir_path = Path(dir_path)
    for img_path in dir_path.glob('*.jpg'):
        images_list.append(str(img_path))
        labels_list.append(int(str(img_path)[-6:-4]))
    return images_list, labels_list

def get_FGNET_images_and_labels(dir_path):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''
    labels_list = []
    images_list = []
    dir_path = Path(dir_path)
    for img_path in dir_path.glob('*.jpg'):
        images_list.append(str(img_path))
        labels_list.append(int(str(img_path)[47:49]))
    return images_list, labels_list

class MORPHDataset(Dataset):
    def __init__(self, dir_path='G:\\vision GNN code\data\morph\Images_ori', transform=MORPH_transform):
        self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        self.images, self.labels = get_MORPH_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

class FGNETDataset(Dataset):
    def __init__(self, dir_path='G:\\vision GNN code\data\FGNET\FGNET\images', transform=FGNET_transform):
        self.dir_path = dir_path  # 数据集根目录
        self.transform = transform
        self.images, self.labels = get_FGNET_images_and_labels(self.dir_path)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100)

