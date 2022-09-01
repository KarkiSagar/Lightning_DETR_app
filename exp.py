import numpy as np 
import pandas as pd 

#Torch
import torch, random, os, pdb
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl

#sklearn
from sklearn.model_selection import StratifiedKFold

#CV
import cv2

import sys
sys.path.append('/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/detr')

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

#Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import warnings
warnings.filterwarnings('ignore')

import wandb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def display_(path,n_folds=5,seed=42):
    marking = pd.read_csv(path)
    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
    marking['x_max']=marking['x']+marking['w']
    marking['y_max']=marking['y']+marking['h']
    
    return df_folds,marking

def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                               
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      
                      A.ToGray(p=0.01),
                      
                      A.HorizontalFlip(p=0.5),
                      
                      A.VerticalFlip(p=0.5),
                      
                      A.Resize(height=512, width=512, p=1),
                      
                      A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      
                      ToTensorV2(p=1.0)],
                      
                      p=1.0,
                     
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
                      )

class WheatDataset(Dataset):
    def __init__(self,image_ids,dataframe,transforms=None,DIR_TRAIN='/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/train'):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms
        self.DIR_TRAIN=DIR_TRAIN
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def get_img(self,image_id):
        path=f'{self.DIR_TRAIN}/{image_id}.jpg'
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image
    
    def get_boxes(self,records,format_='coco'):
         # DETR takes in data in coco format 
        boxes = records[['x', 'y', 'w', 'h']].values
        return boxes
    
    def __getitem__(self,index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        image = self.get_img(image_id)
        
        # DETR takes in data in coco format 
        boxes = self.get_boxes(records)
        
        #Area of bb
        area = boxes[:,2]*boxes[:,3]
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels =  np.zeros(len(boxes), dtype=np.int32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
            
        #Normalizing BBOXES
        _,h,w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels,dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target#, image_id

def collate_fn(batch):
    return tuple(zip(*batch))

class DETRModel(nn.Module):
    def __init__(self,num_classes,num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)

class classifier(pl.LightningModule):
    def __init__(
        self,
        ds,
        bs,
        df,
        df_folds,
        model,
        c,
        run_,
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
    
    def log_boxes_valid(self,images,outputs):
        _,h,w=images[0].shape
        oboxes = outputs['pred_boxes'].detach().cpu().numpy()
        oboxes = [np.array(box) for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes,h,w)]
        wandb_imgs=[]
        for img,boxes in zip(images,oboxes):
            wandb_imgs.append(self.wandb_bbox(img,boxes))
        self.run_.log({"preds": wandb_imgs[:3]})
 
    def wandb_bbox(self,image, bboxes):
        all_boxes = []
        for bbox in bboxes:
            box_data = {"position": {
                            "minX": bbox[0].astype('float'),
                            "minY": bbox[1].astype('float'),
                            "maxX": bbox[0].astype('float')+bbox[2].astype('float'),
                            "maxY": bbox[1].astype('float')+bbox[3].astype('float')
                        },
                         "class_id" : int(0),
                         "box_caption": "Wheat",
                         "domain" : "pixel"}
            all_boxes.append(box_data)
        
        pdb.set_trace()

        return wandb.Image(image, boxes={
            "prediction": {
                "box_data": all_boxes,
              "class_labels": {0:"Wheat"}
            }
        })
            
    def training_step(self,batch,batch_idx):
        images,targets=batch
        images = list(images)
        self.criterion.train()
        outputs=self.model(images)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.run_.log({"train": {"loss":losses}},commit=True)
        return losses
    
    def validation_step(self,batch,batch_idx):
        images,targets=batch
        images=list(images)
        self.criterion.eval()
        outputs=self.model(images)
        self.log_boxes_valid(images,outputs)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        self.run_.log({"val": {"val_loss":losses}},commit=False)
        return losses
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.LR)

def main():
    train_df_path='/nfs/hpc/share/karkisa/Thesis (Action_Seg)/Research paper implementations/Wheat_Detection/train.csv'
    seed=42
    seed_everything(seed)
    fold_df,markings=display_(train_df_path,seed=seed)
    bs=64
    fold=2
    model=DETRModel(num_classes=2,num_queries=100)
    c=SetCriterion(1, matcher=HungarianMatcher(), weight_dict={'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality']).to('cuda')
    run_ = wandb.init(
                        project='e3e3',
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
    
    Trainer=pl.Trainer(devices=1, accelerator="gpu",
                       max_epochs=35,
                      )
    Trainer.fit(Classifier)
    
main()