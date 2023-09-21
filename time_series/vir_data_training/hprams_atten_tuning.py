import os
import json
import plotly.io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import torch


def get_bt(FL, BD):
    while True:
        k_sample = np.random.normal(loc=0.1, scale=0.001)
        relative_k = (k_sample - 0.1) / 0.1
        if -0.1 <= relative_k <= 0.1:
            break
    return FL * BD * k_sample

def von_bertalanffy(t, l_inf=85, l_0=4.0, k=0.1):
    return l_inf - (l_inf - l_0) * np.exp(-k * t)

def generate_random_curves(n, t, l_inf_range=(84, 86), l_0_range=(3.8, 4.2), k_range=(0.095, 0.105)):
    fl_curves = []
    for _ in range(n):
        l_inf = np.random.uniform(*l_inf_range)
        l_0 = np.random.uniform(*l_0_range)
        k = np.random.uniform(*k_range)
        fl_curve = von_bertalanffy(t, l_inf, l_0, k)
        fl_curves.append(fl_curve)
    return np.array(fl_curves)

def generate_noise_curve(mean_curve, noise_scale=1.0, relative_noise_limit=0.1, incremental_noise=0.3):
    noise_curve = np.zeros_like(mean_curve)
    
    for i in range(len(mean_curve)):
        while True:
            if i == 0:
                noise_curve[i] = mean_curve[i]
                break
            
            noise_sample = np.random.normal(loc=mean_curve[i], scale=noise_scale)
            relative_noise = (noise_sample - mean_curve[i]) / mean_curve[i]
            
            if -relative_noise_limit <= relative_noise <= relative_noise_limit:
                if noise_sample > noise_curve[i-1]:
                    noise_curve[i] = noise_sample
                    break
                else:
                    noise_curve[i] = noise_curve[i-1] + np.random.uniform(0, incremental_noise)
                    break
    return noise_curve

def generate_predicted_weight(FL, BD, k=0.001, noise_scale=0.001):
    bt = get_bt(FL, BD)
    predicted_weight = FL * BD * np.pi * k * bt
    noise_weights = np.zeros_like(predicted_weight)

    for i in range(len(predicted_weight)):
        while True:
            if i == 0:
                noise_weights[i] = predicted_weight[i]
                break
            
            noise_sample = np.random.normal(loc=predicted_weight[i], scale=noise_scale)
            relative_noise = (noise_sample - predicted_weight[i]) / predicted_weight[i]
            
            if -0.1 <= relative_noise <= 0.1:
                if noise_sample > noise_weights[i-1]:
                    noise_weights[i] = noise_sample
                    break
                else:
                    noise_weights[i] = noise_weights[i-1] + np.random.uniform(0, 0.3)
                    break
    return predicted_weight, noise_weights

class ModelConfig:
    def __init__(self):
        self.l_inf_range = (70, 90)
        self.l_0_range = (5, 7)
        self.k_range = (0.5, 0.6)
        
        self.fl_noise_config = {
            "noise_scale": 0.01, 
            "relative_noise_limit": 0.1, 
            "incremental_noise": 0.01
        }
        
        self.bd_noise_config = {
            "noise_scale": 0.001, 
            "relative_noise_limit": 0.1, 
            "incremental_noise": 0.001
        }
        
        self.weight_noise_scale = 0.1
        self.trail_num = 0
        self.training_count = 0
        
    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.__dict__))
    
    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = eval(f.read())
            self.__dict__.update(data)
            
def generate_data_and_dataframe(config):
    time_days = np.linspace(0, 410, 482) + 400
    time_years = time_days / 365  

    fl_curves = generate_random_curves(10, time_years, config.l_inf_range, config.l_0_range, config.k_range)
    mean_fl_curve = np.mean(fl_curves, axis=0)
    fl_noise = generate_noise_curve(mean_fl_curve, **config.fl_noise_config)

    bd_curves = generate_random_curves(10, time_years, config.l_inf_range, config.l_0_range, config.k_range)
    bd_curves = bd_curves * 0.169599
    mean_bd_curve = np.mean(bd_curves, axis=0)
    bd_noise = generate_noise_curve(mean_bd_curve, **config.bd_noise_config)

    result_weights, noise_weight = generate_predicted_weight(fl_noise, bd_noise, noise_scale=config.weight_noise_scale)

    data = {
        "Time_Days": time_days[:-2],
        "Time_Years": time_years[:-2],
        "FL_Noise": fl_noise[:-2],
        "BD_Noise": bd_noise[:-2],
        "Result_Weights": result_weights[:-2],
        "Noise_Weight": noise_weight[:-2],
        "Mean_FL_Curve": mean_fl_curve[:-2],  # 원래 곡선 추가
        "Mean_BD_Curve": mean_bd_curve[:-2]   # 원래 곡선 추가
    }
    df = pd.DataFrame(data)
    df['time_index'] = df.index
    df['group'] = 1
    
    return df

def generate_plot(config, df, save_path):
    dot_size = 2

    fig = make_subplots(rows=3, cols=1, subplot_titles=('FL Growth Curve', 'BD Growth Curve', 'Weight Growth Curve'))
    
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["Mean_FL_Curve"], mode='lines', name='Original FL Growth Curve', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["FL_Noise"], mode='lines + markers', name='FL Growth Curve with Noise', marker=dict(color='red', size=dot_size)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["Mean_BD_Curve"], mode='lines', name='Original BD Growth Curve', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["BD_Noise"], mode='lines + markers', name='BD Growth Curve with Noise', marker=dict(color='red', size=dot_size)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["Result_Weights"], mode='lines', name='Weight Growth Curve with Noise', marker=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Time_Days"], y=df["Noise_Weight"], mode='markers', name='Weight Growth Curve with Noise', marker=dict(color='red', size=dot_size)), row=3, col=1)
    
    fig.update_layout(title='Growth Curves with Noise', height=1200, width=1000)

    # Create directories if not exist and save the plot as HTML
    os.makedirs(save_path, exist_ok=True)
    plotly.io.write_html(fig, os.path.join(save_path, 'growth_curves_with_noise.html'))

    
    
    
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

def atten_tuning(atten_head):
    text_path = 'config_atten_head.txt'
    config = ModelConfig()
    config.load_from_file(text_path)

    # model prediction interval setting
    max_prediction_length = 7
    max_encoder_length = 30

    # You will need to define or import the generate_data_and_dataframe function
    data = generate_data_and_dataframe(config)  
    train_data = data[data['time_index'] < 384 - max_prediction_length]
    test_data = data[data['time_index'] >= 384 - max_prediction_length]


    save_path = f"results_tuning/atten_head/trail_{config.trail_num}/{config.training_count}"
    generate_plot(config, data, save_path)

    data = data[['FL_Noise' , 'BD_Noise' , 'Noise_Weight' , 'time_index' , 'group']]

    training = TimeSeriesDataSet(
        train_data , 
        time_idx="time_index",
        target="Noise_Weight",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['FL_Noise' ,'BD_Noise'],
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation=None
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


    validation = TimeSeriesDataSet.from_dataset(training, test_data, predict=False, stop_randomization=True)
    print(f'training length {len(training)} , test length {len(validation)}')
    # create dataloaders for model
    batch_size = 64  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=10, verbose=False, mode="min")
    logger = TensorBoardLogger(f"lightning_logs/trail_config_hparams_atten_head_tuning/{config.training_count}")

    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=500,
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
        learning_rate=1e-2,
        hidden_size=32,
        attention_head_size=atten_head,
        dropout=0.1,
        hidden_continuous_size=32,
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
    

    config.training_count += 1
    config.save_to_file(text_path)

    nrmse_list = []
    figs = []

    pred_save_path = save_path
    raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True , trainer_kwargs=dict(accelerator="cpu"))
    for idx in range(len(raw_predictions.x['decoder_target'])):
        real = raw_predictions.x['decoder_target'][idx]
        pred = raw_predictions.output['prediction'][idx, :, 3]

        mse = torch.mean((real - pred)**2)
        rmse = torch.sqrt(mse)
        norm_rmse = rmse / (torch.max(real) - torch.min(real))  # 1e-8을 추가하여 0으로 나누기를 방지

        nrmse_list.append(norm_rmse.item())
        fig = tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        figs.append(fig)

    nrmse_df = pd.DataFrame(nrmse_list, columns=['NRMSE'])
    with PdfPages(f'{pred_save_path}/all_plots.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)  # 생성된 fig 객체를 닫습니다
    nrmse_df.to_csv(f'{pred_save_path}/nrmse_results.csv', index=False)
    
for atten_head in [1,2,3,4,5,6,7,8,9]:
    atten_tuning(atten_head)

