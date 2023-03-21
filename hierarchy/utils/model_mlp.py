# Загрузка библиотек
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score
from utils.lion_pytorch import Lion


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size=300, output_size=874):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class ClassifierMLP(pl.LightningModule):
    # Класс классификационной модели, регулирует все параметры
    def __init__(self,
                 model_hparams,
                 optimizer_params=None,
                 scheduler=False,
                 ):

        super().__init__()
        # сохранение гиперпараметров
        self.save_hyperparameters()
        # классфиикатор в конце default 1024 + 300 -> 874
        self.classifier = MultilayerPerceptron()
        # размер батча
        self.batch_size = self.hparams.model_hparams['batch_size']
        # определение функции ошибки
        self.criterion = nn.CrossEntropyLoss()
        # определение метрики
        self.fscore = F1Score(task="multiclass", average='macro',
                              num_classes=self.hparams.model_hparams['num_classes'])

    def forward(self, x):
        out = self.classifier(x.float())
        return out

    def configure_optimizers(self):
        # конфигурация оптимизатора
        # print(list(self.named_parameters()))
        parameters = nn.ParameterList(self.parameters())

        trainable_parameters = nn.ParameterList(
            filter(lambda p: p.requires_grad, parameters))

        if self.hparams.optimizer_params['name'] == 'Lion':
            optimizer = Lion(trainable_parameters,
                             self.hparams.optimizer_params['lr'])
        elif self.hparams.optimizer_params['name'] == "Adam":
            optimizer = torch.optim.Adam(
                trainable_parameters, self.hparams.optimizer_params['lr'])
        elif self.hparams.optimizer_params['name'] == "AdamW":
            optimizer = torch.optim.AdamW(
                trainable_parameters, self.hparams.optimizer_params['lr'])
        elif self.hparams.optimizer_params['name'] == "RAdam":
            optimizer = torch.optim.RAdam(
                trainable_parameters, self.hparams.optimizer_params['lr'])
        elif self.hparams.optimizer_params['name'] == "SGD":
            optimizer = torch.optim.SGD(
                trainable_parameters, self.hparams.optimizer_params['lr'])
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_params["name"]}"'

        # конфигурация расписания изменений для шага обучения
        if self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=True)
            return [optimizer], [scheduler]
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 15)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.classifier(X.float())
        loss = self.criterion(preds, y)
        _, predicted = torch.max(preds.data, 1)
        acc = self.fscore(predicted, y)
        # логирование результатов
        self.log('train_loss', loss, on_step=True, logger=True,)
        self.log('train_acc', acc, on_step=True, logger=True,)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.classifier(X.float())
        loss = self.criterion(preds, y)
        _, predicted = torch.max(preds.data, 1)
        acc = self.fscore(predicted, y)
        # логирование результатов
        self.log('val_loss', loss, on_step=False, logger=True,)
        self.log('val_acc', acc, on_step=False, logger=True,)
        return loss

