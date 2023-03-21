import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from coca_pytorch.coca_pytorch import CoCa
from torchmetrics import F1Score
from utils.lion_pytorch import Lion
from vit_pytorch.extractor import Extractor
from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT


class ClassifierCOCA(pl.LightningModule):
    # Класс классификационной модели, регулирует все параметры
    def __init__(self,
                 model_hparams,
                 optimizer_params=None,
                 scheduler=False,
                 ):

        super().__init__()
        # сохранение гиперпараметров
        self.save_hyperparameters()
        vit = SimpleViT(
            image_size=model_hparams['image_size'],
            patch_size=32,
            num_classes=model_hparams['num_classes'],
            dim=model_hparams['len_image_emb'],
            depth=6,
            heads=16,
            mlp_dim=2048,
            patch_dropout=0.5  # https://arxiv.org/abs/2212.00794
        )
        vit = Extractor(vit, return_embeddings_only=True, detach=False)
        self.model = CoCa(
            dim=300,                     # размерность тестового эмбеддинга
            img_encoder=vit,
            # размерность эмбеддинга картинки
            image_dim=model_hparams['len_image_emb'],
            num_tokens=500000,           # размер словаря токенов
            unimodal_depth=6,            # глубина unimodal transformer
            multimodal_depth=6,          # глубина multimodal transformer
            dim_head=64,                 # размерность каждой attention head
            heads=8,                     # количество attention heads
            caption_loss_weight=1.,      # weight on the autoregressive caption loss
            # weight on the contrastive loss between image and text CLS embeddings
            contrastive_loss_weight=1.,
        )
        # размер батча
        self.batch_size = self.hparams.model_hparams['batch_size']
        # определение функции ошибки

    def forward(self, img, text):
        text_embeds, image_embeds = self.model(text=text,
                                               images=img,
                                               return_embeddings=True)
        return text_embeds, image_embeds

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
                optimizer, 10)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        image, text = batch
        loss = self.model(text=text,
                          images=image,
                          return_loss=True,  # set this to True to get the full caption + contrastive loss
                          )
        # логирование результатов
        self.log('train_loss', loss, on_step=True,
                 logger=True, batch_size=self.batch_size)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        image, text = batch
        loss = self.model(text=text,
                          images=image,
                          return_loss=True,  # set this to True to get the full caption + contrastive loss
                          )
        # логирование результатов
        self.log('val_loss', loss, on_step=True,
                 logger=True, batch_size=self.batch_size)
        return loss
