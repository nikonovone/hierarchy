# загрузка библиотек
import ast
import re

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.model_coca import ClassifierCOCA

from utils.model_mlp import ClassifierMLP


# ---------------------TRAIN MULTILAYER PERCEPTRON---------------------
def train_model_mvp(**params):
    """Обучение модели

    """
    # создание тренера
    trainer = pl.Trainer(
        default_root_dir=params['checkpoint_path'],
        accelerator='gpu' if str(params['device']) == 'cuda' else 'cpu',
        devices=params['gpu_id'],
        max_epochs=params['num_epochs'],
        logger=params['logger'],
        log_every_n_steps=1,
        callbacks=[
            LearningRateMonitor('step'),
            RichProgressBar(theme=RichProgressBarTheme()),
            # FeatureExtractorFreezeUnfreeze(
            #     unfreeze_at_epoch=params['unfreeze_epoch']),
            ModelCheckpoint(dirpath=params['checkpoint_path'],
                            save_weights_only=False,
                            save_top_k=3,
                            mode='min',
                            monitor='val_loss',
                            filename='{epoch}-{val_loss:.5f}'
                            ),
        ],
    )
    # установка параметров логгирования
    if params['logger']:
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None

    # для репродукции обучения
    pl.seed_everything(params['seed'])
    # создание модели
    model = ClassifierMLP(params['model_hparams'],
                          params['optimizer_params'],
                          params['scheduler'])
    # обучение модели
    trainer.fit(model, params['data_module'], ckpt_path=params['ckpt_path'])
    return trainer


# ---------------------TRAIN COCA MODEL---------------------


def train_model_coca(**params):
    """Обучение модели

    """
    # создание тренера
    trainer = pl.Trainer(
        # директория для чекпоинтов
        default_root_dir=params['checkpoint_path'],
        # выбор устройства для обучения
        accelerator='gpu' if str(params['device']) == 'cuda' else 'cpu',
        # номер GPU
        devices=params['gpu_id'],
        # количество эпох
        max_epochs=params['num_epochs'],
        # логер
        logger=params['logger'],
        # шаг логирования
        log_every_n_steps=1,
        # установка callbacks
        callbacks=[
            LearningRateMonitor('step'),
            RichProgressBar(theme=RichProgressBarTheme()),
            # FeatureExtractorFreezeUnfreeze(
            #     unfreeze_at_epoch=params['unfreeze_epoch']),
            ModelCheckpoint(
                dirpath=params['checkpoint_path'],
                save_weights_only=False,
                # сохранять топ-k чекпоинты
                save_top_k=3,
                # режим определения лучшего скора
                mode='min',
                # какую метрику отслеживать
                monitor='val_loss',
                # имя чекпоинта
                filename='{epoch}-{val_loss:.5f}'
            ),
        ],
    )
    # установка параметров логгирования
    if params['logger']:
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = None

    # для репродукции обучения
    pl.seed_everything(params['seed'])
    # создание модели
    model = ClassifierCOCA(params['model_hparams'],
                           params['optimizer_params'],
                           params['scheduler'])
    # обучение модели
    trainer.fit(model, params['data_module'], ckpt_path=params['ckpt_path'])
    return trainer


# ---------------------PREPROCESSING TEXT---------------------

def cleanhtml(raw_html: str):
    """Очищает строк от html тегов

    Args:
        raw_html (str): строка для обработки

    Returns:
        str: обработанная строка
    """
    cleantext = re.sub(re.compile('<.*?>'), '', raw_html)
    cleantext = ast.literal_eval(cleantext)
    return cleantext


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


def lemmatize(text: str, navec) -> list:
    """Лемматизация и кодирвока из словаря Navec строки

    Args:
        text (_type_): строка для обработки
        navec (): экзмепляр navec.Navec()

    Returns:
        list: список из закодированной строки
    """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    lemmas = [token.lemma for token in doc.tokens]
    tokens = [navec.vocab[token] for token in lemmas if token in navec]
    return tokens


def transform_tokens(tokens: list, max_size=300) -> torch.Tensor:
    """Добавялет паддинг и преобразует список в тензор

    Args:
        tokens (list): список с закодированными токенами

    Returns:
        torch.Tensor: тензор с закодирвоанными токенами с фиксированной длиной
    """

    if not isinstance(tokens, list):
        tokens = ast.literal_eval(tokens)
    text = torch.Tensor(tokens).long()
    text = np.pad(text, pad_width=(
        0, max_size-text.shape[0]), mode='constant', constant_values=0)
    return text


def predict(df, model, images_path, device):
    """Формирование предсказаний модели

    Args:
        df (_type_): _description_
        model (_type_): _description_
        images_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    with torch.no_grad():

        text_embeddings = []
        image_embeddings = []
        for _, image_name, tokens in tqdm(df[['product_id', 'title']].itertuples(), total=len(df)):
            path_img = str(images_path / str(image_name)) + '.jpg'
            img = cv2.cvtColor(cv2.imread(path_img),
                               cv2.COLOR_BGR2RGB).transpose(2, 0, 1)/255
            img = torch.from_numpy(img).unsqueeze(0).float().to(device)

            text = torch.Tensor(tokens).unsqueeze(0).long().to(device)

            text_embeds, image_embeds = model(img, text)
            text.cpu()
            text.cpu()
            text_embeddings.append(text_embeds.cpu().squeeze())
            image_embeddings.append(image_embeds.cpu().squeeze())

    return torch.stack(image_embeddings).cpu(), torch.stack(text_embeddings).cpu()


def normalization_df(img_emb: torch.Tensor, txt_emb: torch.Tensor):
    scaler_img = StandardScaler()
    scaler_text = StandardScaler()
    image_values = scaler_img.fit_transform(img_emb.numpy())
    text_values = scaler_text.fit_transform(txt_emb.numpy())

    total_embs = image_values + text_values

    return total_embs
