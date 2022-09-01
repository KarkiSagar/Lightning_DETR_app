import numpy as np 

#Torch
import torch, pdb
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#Albumenatations
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

import wandb
from Pipeline import get_train_transforms,get_valid_transforms
import  torchmetrics
# p=torchmetrics.detection.mean_ap.MeanAveragePrecision
def collate_fn(batch):
    return tuple(zip(*batch))

class classifier(pl.LightningModule):
    def __init__(
        self,
        ds,
        bs,
        df,
        df_folds,
        model,
        c,
        run_=None,
        fold=1,
        LR=2e-5,
    ):
        super().__init__()
        self.ds=ds
        self.bs=bs
        self.df=df
        self.train_img_ids,self.val_img_ids=df_folds[df_folds['fold'] != fold].index.values, df_folds[df_folds['fold'] == fold].index.values
        self.model=model
        self.LR=LR
        self.criterion = c
        self.run_=run_
        
    def train_dataloader(self):
        train_ds=self.ds(self.train_img_ids,self.df,transforms=get_train_transforms())
        train_loader=DataLoader(train_ds,batch_size=self.bs,shuffle=False,
                                num_workers=4,
                                collate_fn=collate_fn)
        return train_loader
      
    def val_dataloader(self):
        val_ds=self.ds(self.val_img_ids,self.df,transforms=get_valid_transforms())
        val_loader=DataLoader(val_ds,batch_size=self.bs,shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
        return val_loader

    def get_one_pred_record(self,images,outputs):

        _,h,w=images[0].shape
        oboxes = outputs['pred_boxes'].detach().cpu().numpy()
        oboxes = [np.array(box) for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes,h,w)]

        boxes_dict={
                        "predictions": {
                        "box_data": [{
                            # one box expressed in the default relative/fractional domain
                            "position": {
                                "minX": box[0].astype('float'),
                                "minY": box[1].astype('float'),
                                "maxX": box[0].astype('float')+box[2].astype('float'),
                                "maxY": box[1].astype('float')+box[3].astype('float')
                            },
                            "class_id" : 0,
                            "box_caption": "Wheat",
                        } for box in oboxes[0]
                        ]}}

        img=wandb.Image(images[0], boxes=boxes_dict)

        self.run_.log({"pred_img":img})
        return 
            
    def training_step(self,batch,batch_idx):
        images,targets=batch
        images = list(images)
        self.criterion.train()
        outputs=self.model(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if self.run_:
            self.run_.log({"train": {"loss":losses}},commit=True)
        return losses
    
    def validation_step(self,batch,batch_idx):
        images,targets=batch
        images=list(images)
        self.criterion.eval()
        outputs=self.model(images)
        # pdb.set_trace()
        self.get_one_pred_record(images,outputs)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if self.run_:
            self.run_.log({"val": {"val_loss":losses}},commit=False)
        return losses
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.LR)


