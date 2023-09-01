#!/usr/bin/env python
# coding: utf-8

import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0 , 1"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import random
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, feature_extractor, files):
        self.feature_extractor = feature_extractor
        self.files = files

        self.id2label = {0: "background", 1: "fish"}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]['image'])
        segmentation_map = Image.open(self.files[idx]['label'])
        segmentation_array = np.array(segmentation_map)

        segmentation_array = np.max(segmentation_array, axis=-1)
        # Convert 255 values to 1
        segmentation_array = segmentation_array // 255

        segmentation_map = Image.fromarray(segmentation_array)
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs


# In[6]:


from transformers import SegformerFeatureExtractor
pre_trained = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
feature_extractor = SegformerFeatureExtractor.from_pretrained(pre_trained,
                                                             do_normalize  = False)

feature_extractor.size = 512


import os
import random
from torch.utils.data import DataLoader, random_split

# File paths for each category
original_base_path = '../Dataset/salmon_segmentation/out/1.원천데이터'
label_base_path = '../Dataset/salmon_segmentation/out/2.라벨링데이터'
categories = ['01-치어-송어사료급이', '02-치어-연어사료급이', '03-성어-송어사료급이', '04-성어-연어사료급이']

# Accumulate file names from each category
paired_files = []
for category in categories:
    path_original = os.path.join(original_base_path, category)
    path_label = os.path.join(label_base_path, category)

    original_files = sorted([os.path.join(path_original, f) for f in os.listdir(path_original) if f.endswith(('.jpg', '.png'))])
    label_files = sorted([os.path.join(path_label, f) for f in os.listdir(path_label) if f.endswith('.png')])

    paired_files.extend([{"image": img, "label": lbl} for img, lbl in zip(original_files, label_files)])

random.seed(42)
random.shuffle(paired_files)

num_train = int(0.8 * len(paired_files))
num_val = int(0.1 * len(paired_files))
num_test = len(paired_files) - num_train - num_val

train_files, val_files, test_files = random_split(paired_files, [num_train, num_val, num_test])

train_dataset = SegmentationDataset(feature_extractor, train_files)
val_dataset = SegmentationDataset(feature_extractor, val_files)
test_dataset = SegmentationDataset(feature_extractor, test_files)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from datasets import load_metric

class SegformerFinetuner(pl.LightningModule):    
    def __init__(self, id2label, pre_trained ,  train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        print(f"pre_trained_mode : {pre_trained}")
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        
        print(self.id2label)
        print(self.label2id)
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pre_trained, 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
                
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            
            for k,v in metrics.items():
                self.log(k,v)
            
            return(metrics)
        else:
            return({'loss': loss})
    
    def validation_step(self, batch, batch_nb):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255, 
            reduce_labels=False,
        )
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        val_background_iou = metrics['per_category_iou'][0]
        val_foreground_iou = metrics['per_category_iou'][1]
        
        self.log("val_mean_iou", val_mean_iou, prog_bar=True, logger=True)
        self.log("val_background_iou", val_background_iou, prog_bar=True, logger=True)
        self.log("val_foreground_iou", val_foreground_iou, prog_bar=True, logger=True)
        self.log("val_mean_accuracy", val_mean_accuracy, prog_bar=True, logger=True)
        
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        return({'test_loss': loss})
    
    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"test_loss": avg_test_loss, 
                   "test_mean_iou":test_mean_iou, 
                   "test_mean_accuracy":test_mean_accuracy,
                   "backgournd_iou" : metrics['per_category_iou'][0] , 
                   "foregournd_iou" : metrics['per_category_iou'][1]
                   }
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl


# In[14]:


from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import torch
import torch
from torch import nn


# In[15]:


id2label = {0: "background", 1: "fish"}

segformer_finetuner = SegformerFinetuner(
    id2label, 
    pre_trained = pre_trained,
    train_dataloader=train_dataloader, 
    val_dataloader=val_dataloader, 
    test_dataloader=test_dataloader, 
    metrics_interval=10,
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="min",
)

pl.seed_everything(42)
trainer = pl.Trainer(
    devices=2, 
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=500,
    #accelerator="gpu"
    strategy='ddp'
)
print('b')

trainer.fit(segformer_finetuner)
torch.save(segformer_finetuner.model , './model_save/2023_09_01_salmon_data_init.pt')


# In[ ]:




