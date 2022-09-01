---

<div align="center">    
 
# Quick Prototyping Object Detection Research Paper    


</div>
 
 
## Description   
Object Detection on any data set using supervised learning.
End to end architecture using Pytorch Lightning. 
Compatible with any object detection model as long as they are in pytorch and have a loss/training logic.
[Monitoring training](https://wandb.ai/karkisa/Super_Enigma/runs/3apwmdvb?workspace=user-karkisa) using Weights and Bias.


Right now the repo trains DETR model on [Wheat head detection data](https://www.kaggle.com/competitions/global-wheat-detection/data)

Initial [Kaggle Notebook](https://www.kaggle.com/code/karkisa/small-object-detection-using-pytorch-lightning)

## How to run {example}
First, install dependencies   
```bash
# clone project   
git clone https://github.com/karkisa/super-enigma.git
# install project   
cd super-enigma 
pip install -e .   
pip install -r requirements.txt
kaggle competitions download -c global-wheat-detection      # get data from kaggle
unzip global-wheat-detection.zip                            # unzip the data 

# clone your model's repo
!git clone https://github.com/facebookresearch/detr.git  -q  # used for loss function , architecture and training logic
 ```   

Change the paths based on paths on your machine
# Research Playground
python Playground.py    
```
def main():
    #train_df_path on your 
    train_df_path='train.csv'
    seed=42
    seed_everything(seed)
    fold_df,markings=display_(train_df_path,seed=seed)
    bs=64
    fold=2
    model=DETRModel(num_classes=2,num_queries=100)
    c=SetCriterion(1, matcher=HungarianMatcher(), weight_dict={'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}, eos_coef = 0.5, losses=['labels', 'boxes', 'cardinality']).to('cuda')
    run_ = wandb.init(
                        project='super-enigma',
                        group=str(fold),
                        name='exp1'
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
```

### Citation   
```
@article{Sagar Karki,
  title={Super Sonic Speed Prototyping Object Detection Research Rapers},
  author={Sagar Karki},
  year={2022}
}
```   
