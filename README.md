# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

I have upgraded the origin repository with the following changes:

Added Weights & Biases (WandB) integration.
Added YAML configuration support.
Implemented linear warmup for the learning rate.
Added early stopping (training dataset split into train and validation for this feature).
Changed the optimizer from Adam to RAdam.
Hyperparameters can be adjusted in the /config_yaml file.
The repository has been tested with PyTorch 2.4.

I used the Docker image:
pytorch:2.4.1-cuda12.4-cudnn9-devel
Docker Hub: Image Details

Notes:
cuDNN Version: The Docker image does not include cuDNN. I manually downloaded and installed cuDNN 9. Please ensure the correct version is installed.
Weights & Biases:
If you do not have a WandB account, follow the Quickstart Guide to create one.
Retrieve your API key from this link.


# Download data and running

```
git clone https://github.com/hunsoo0823/pointnet.pytorch.git
cd pointnet.pytorch
pip install -r requirements.txt
```

Download and build visualization tool
```
cd scripts
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils
python train_classification.py
python train_segmentation.py
```

Use `--feature_transform` to use feature transform.

# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | 89.2 | 
| this implementation(w/o feature transform) | 86.4 | 
| this implementation(w/ feature transform) | 87.0 | 

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc | 
| :---: | :---: | 
| Original implementation | N/A | 
| this implementation(w/o feature transform) | 98.1 | 
| this implementation(w/ feature transform) | 97.7 | 

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:
![seg](https://github.com/hunsoo0823/pointnet.pytorch/blob/master/misc/show3d.png)

Wandb result:
![wandb](https://github.com/hunsoo0823/pointnet.pytorch/blob/master/misc/wandb.png)


thank you! fxia22 i learn a lot in your code

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)

## To-Do List
- [ ] Update new performacne
- [ ] Add how to visualization