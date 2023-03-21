# Загрузка библиотек
import ast

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    # Класс датасета
    def __init__(self, data, data_path, transform=None):
        self.data = data
        self.data_path = data_path
        self.images = data['product_id']
        self.text = data['title']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Чтение изображения
        path_img = str(self.images.iloc[index])
        img = cv2.cvtColor(cv2.imread(
            str(self.data_path / path_img) + '.jpg'), cv2.COLOR_BGR2RGB)

        text = torch.from_numpy(self.text[index]).long()

        # Применение трансформации и нормализация
        if self.transform:
            transform = self.transform(image=img)
            img = (transform['image']/255).float()
        else:
            img = torch.Tensor(img)/255
        return img, text


class DataModule(pl.LightningDataModule):
    """Описывает логику работу с данными.
    """

    def __init__(self,
                 df,
                 data_path,
                 batch_size=128,
                 num_workers=4,
                 val_size=0.1,
                 image_size = 256,
                 seed=13):
        super().__init__()
        # dataframe
        self.df = df
        self.data_path = data_path
        # число используемых ядер
        self.num_workers = num_workers
        # размер батча
        self.batch_size = batch_size
        # размер валидационной выборки
        self.val_size = val_size
        # инициализационное зерно дял псевдослучайных процессов
        self.seed = seed

        self.transform = A.Compose([
            A.Resize(image_size,image_size),
            ToTensorV2(),
        ])
    # подготовка и разделение данных

    @staticmethod
    def my_collate_fn(data):
        return tuple(data)

    def setup(self, stage=None):
        train, val = train_test_split(
            self.df, test_size=self.val_size, random_state=self.seed)

        self.train_dataset = CustomDataset(train.reset_index(drop=True), self.data_path, self.transform)
        self.val_dataset = CustomDataset(val.reset_index(drop=True), self.data_path, self.transform)

    def train_dataloader(self):
        # создание тренировачного даталоадера
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=self.my_collate_fn)

    def val_dataloader(self):
        # создание валидационного даталоадера
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True,
                          collate_fn=self.my_collate_fn)