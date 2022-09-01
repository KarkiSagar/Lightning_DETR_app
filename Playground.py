#Torch
import pdb
import pytorch_lightning as pl

import sys
sys.path.append('detr')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import warnings
warnings.filterwarnings('ignore')

import wandb
from Pipeline import seed_everything, display_,WheatDataset
from Model import DETRModel
from Logic import classifier

# from torchmetrics.detection.mean_ap import MeanAveragePrecision
def main():
    # import  torchmetrics
    # p=detection.mean_ap.MeanAveragePrecision
    train_df_path='data/train.csv'
    seed=42
    seed_everything(seed)
    fold_df,markings=display_(train_df_path,seed=seed)
    bs=64
    fold=2
    model=DETRModel(num_classes=2,num_queries=100)
    c=SetCriterion(1, matcher=HungarianMatcher(), weight_dict={'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality'])#.to('cuda')
    run_ = wandb.init(
                        project='ee2',
                        group=str(fold),
                        name='exp13'
                    )

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
    
    Trainer=pl.Trainer(devices=1, accelerator="cpu",
                       max_epochs=35,
                      )
    Trainer.fit(Classifier)

if __name__=="__main__":
    main()