# -*- coding: utf-8 -*-
import os
import copy
import time
import json
import torch
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss

from IPython.core.display import HTML
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import plotly.graph_objects as go


def batch_tuning(batch_size):

    data = pd.read_csv('../Dataset/dariy_data/cow_weight_data.csv')
    data = data[data['time_idx'] < 240]

    max_prediction_length = 7
    max_encoder_length = 30


    training_cutoff = data["time_idx"].max() - (max_prediction_length + max_encoder_length) 
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="체중(kg)",
        group_ids=["개체번호"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=['WOD'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['일일 착유 횟수' , '착유량(L)' , '섭취량(kg)', 'CO2 측정값' , '온도 측정값' , 'age'],
        target_normalizer=GroupNormalizer(
            groups=["개체번호"], transformation=None
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        #allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, data[data['time_idx'] > training_cutoff] , predict=True, stop_randomization=True)

    batch_size = 256  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger(f"lightning_logs/batch_tuning")  # logging results to a tensorboard

    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[early_stop_callback],
        logger=logger,
        devices = [0]
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-4,
        hidden_size=96,
        attention_head_size=6,
        dropout=0.1,
        hidden_continuous_size=96,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True , return_y=True ,  trainer_kwargs=dict(accelerator="cpu"))

    fig_save_path = f'./fig/batch_size_{batch_size}'
    os.makedirs(fig_save_path, exist_ok=True)
    for idx in range(25):  # plot 10 examples
        fig = tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        fig.set_size_inches(14, 6)
        fig.savefig(fig_save_path + f'/{idx}.png')       
        
for batch_size in [32, 48, 64, 78 , 96, 128, 156, 178, 196, 212, 256]:
    batch_tuning(batch_size)
   