# Загрузка библиотек
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset



class CustomDataset(Dataset):
    # Класс датасета
    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index][:-1]
        y = self.data[index][-1:]
        return torch.from_numpy(X).float(), torch.from_numpy(y).squeeze().long()

# Класс модуля данных


class DataModule(pl.LightningDataModule):
    """Описывает логику работу с данными.
    """

    def __init__(self,
                 data,
                 batch_size=128,
                 num_workers=4,
                 val_size=0.1,
                 seed=13):
        super().__init__()
        # информация для обучения
        self.data = data
        # число используемых ядер
        self.num_workers = num_workers
        # размер батча
        self.batch_size = batch_size
        # размер валидационной выборки
        self.val_size = val_size
        # инициализационное зерно дял псевдослучайных процессов
        self.seed = seed


    # подготовка и разделение данных
    def setup(self, stage=None):
        train, val = train_test_split(
            self.data, test_size=self.val_size, random_state=self.seed)

        self.train_dataset = CustomDataset(train)
        self.val_dataset = CustomDataset(val)
        # self.test_dataset = CustomDataset(test)

    def train_dataloader(self):
        # создание тренировачного даталоадера
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        # создание валидационного даталоадера
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)
