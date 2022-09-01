#Torch
import pdb
import pytorch_lightning as pl
import streamlit as st
import sys
sys.path.append('detr')
import torch
from torchvision import transforms
from lightning.app.components.serve import ServeGradio
import numpy as np
import gradio as gr
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import warnings
warnings.filterwarnings('ignore')

import wandb
from Pipeline import seed_everything, display_,WheatDataset
from Model import DETRModel
from Logic import classifier

import joblib
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

from lightning.app.storage import Path
import lightning as L
import lightning.app as L_app




class WebLinkComponent(L.LightningFlow):
    def configure_layout(self):
        tab_1 = {
            "name": "Logger",
            "content": "https://bit.ly/tb-aasae"
        }
        tab_2 = {
            "name": "Paper",
            "content": "https://arxiv.org/pdf/2107.12329.pdf"
        }
        return tab_1, tab_2


class TrainModel(L.LightningWork):

    """This component trains a Sklearn SVC model on digits dataset."""

    def __init__(self):
        super().__init__()
        # 1: Add element to the state.
        self.best_model_path = None
        

    def run(self):
        # get the data csv
        train_df_path='data/train.csv'
        #set seed for reproducibility
        seed=42 
        seed_everything(seed)
        #creating validatin fold
        fold_df,markings=display_(train_df_path,seed=seed)
        fold=2
        #set batch size
        bs=64
        #select model
        model=DETRModel(num_classes=2,num_queries=100)
        #select relative loss fucntion
        c=SetCriterion(1, matcher=HungarianMatcher(), weight_dict={'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality'])#.to('cuda')
        #initiate waanb training log
        run_ = wandb.init(
                            project='ee2',
                            group=str(fold),
                            name='exp13'
                        )
        self._url=run_.get_url()

        #use the clissfier logi for training the model with the specified loss and other parameters

        Classifier=classifier(
            WheatDataset,
            bs,
            markings,
            fold_df,
            model,
            c,
            run_,
            fold=fold
        )
        
        # Initiate the lightning trainer 
        Trainer=pl.Trainer( 
                            devices=1,
                            accelerator="cpu",
                            max_epochs=35,
                            default_root_dir='/Users/sagarkarki/Desktop/Project/Apply1/super-enigma/Trainer_save_dir'
                            )
        
        #Useing Lightning trainier to train the model
        Trainer.fit(Classifier)
        

class LitApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.train_work = TrainModel()
        self.model_demo = ModelDemo()
        
    def run(self) :
        self.model_demo.run()
        self.train_work.run()
        
    def configure_layout(self):
        tab_1 = {
            "name": "Logger",
            "content": self.train_work
        }
        tab_2 = {
            "name": "demo",
            "content": self.model_demo.url
        }
        return tab_2,tab_1

class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.
    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    # inputs = gr.inputs.Textbox(default="Going into the space", label="Unsplash Image Search")
    # outputs = gr.outputs.HTML(label="Images from Unsplash")
    inputs = gr.Image(type='numpy',)
    outputs = gr.Numpy() # gr.HTML(label="Images from Unsplash")

    def __init__(self):
        super().__init__(parallel=True)

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def build_model(self) :
        model=torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)#DETRModel(num_classes=2,num_queries=100)
        return model

    def predict(self, inp):
        # path='/Users/sagarkarki/Desktop/Project/Apply1/super-enigma/Trainer_save_dir/lightning_logs/version_0/checkpoints/epoch=0-step=43.ckpt'
        model=torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)#DETRModel(num_classes=2,num_queries=100)
        inp = transforms.ToTensor()(inp).unsqueeze(0)
        with torch.no_grad():
            # prediction = model(inp)
            outputs = model(inp)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.1
        print( inp.shape[-2:])
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], inp.shape[-2:])    
        
        print(bboxes_scaled.numpy())
        
        return bboxes_scaled


app = L.LightningApp(LitApp())
