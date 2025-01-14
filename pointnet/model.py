from abc import abstractmethod
from typing import Optional
from typing import Dict
from typing import List
from typing import Union
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import random_split
from torchvision import transforms
import wandb
from datetime import datetime
import sys
import os

from utils.data_utils import dataset_split
from utils.config_utils import flatten_dict
from utils.config_utils import register_config
from utils.config_utils import configure_optimizers_from_cfg
from utils.config_utils import get_loggers
from utils.config_utils import get_callbacks
from utils.custom_math import softmax
from pointnet.dataset import ShapeNetDataset


class BaseLightningModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig, is_cls=True, num_seg_classes=-1):
        super().__init__()
        self.cfg = cfg
        self.is_cls = is_cls
        if not self.is_cls:
            self.num_seg_classes = num_seg_classes
        # self.loss_function = nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self, inputs, target):
        raise NotImplemented()  

    def configure_optimizers(self):
        self._optimizer, self._schedulers = configure_optimizers_from_cfg(self.cfg, self)
        return self._optimizer, self._schedulers

    def _forward(self, points, target, mode: str):

        assert mode in ["train", "val", "test"]

        # get predictions
        #outputs = self(images)

        points = points.transpose(2, 1)

        pred, _, _ = self(points)

        if self.is_cls:
            target = target[:, 0]
        else:
            target = target
            pred = pred.view(-1, self.num_seg_classes)
            target = target.view(-1, 1)[:, 0] - 1

        # get loss(calculate Loss)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        corrects = torch.sum(pred_choice == target.data)
        acc = corrects / len(pred)

        return {
            f"{mode}_loss": loss,
            f"{mode}_acc": acc,
        }

    def training_step(self, batch):
        points, target = batch
        logs = self._forward(points, target, mode="train")
        self.log_dict(logs)
        logs["loss"] = logs["train_loss"]
        cul_lr = self._optimizer.param_groups[0]["lr"] if self._schedulers is None else self._schedulers[0].get_last_lr()[0]
        wandb.log({"learning_rate": cul_lr})

        return logs

    def validation_step(self, batch):
        points, target = batch
        logs = self._forward(points, target, mode="val")
        self.log_dict(logs)
        logs["loss"] = logs["val_loss"]
        return logs

    def test_step(self, batch):
        points, target = batch
        logs = self._forward(points, target, mode="test")
        self.log_dict(logs)
        logs["loss"] = logs["test_loss"]
        return logs
    

class PointNetCls(BaseLightningModule):
    def __init__(self, cfg: DictConfig, num_classes=2):
        BaseLightningModule.__init__(self, cfg=cfg, is_cls=True)
        
        self.feat = PointNetfeat(cfg, global_feat=True)
        self.fc1 = nn.Linear(cfg.model.cls.layer1.in_feature, cfg.model.cls.layer1.out_feature)
        self.fc2 = nn.Linear(cfg.model.cls.layer2.in_feature, cfg.model.cls.layer2.out_feature)
        self.fc3 = nn.Linear(cfg.model.cls.layer3.in_feature, k)
        self.dropout = nn.Dropout(cfg.model.cls.dropout)
        self.bn1 = nn.BatchNorm1d(cfg.model.cls.layer1.out_feature)
        self.bn2 = nn.BatchNorm1d(cfg.model.cls.layer2.out_feature)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat
    

class PointNetDenseCls(BaseLightningModule):
    def __init__(self, cfg: DictConfig, num_seg_classes):
        BaseLightningModule.__init__(self, cfg=cfg, is_cls=False, num_seg_classes=num_seg_classes)
        
        self.num_seg_classes = num_seg_classes
        self.feat = PointNetfeat(cfg,global_feat=False)
        self.conv1 = torch.nn.Conv1d(
            cfg.model.densecls.layer1.conv1d_in_channel,
            cfg.model.densecls.layer1.conv1d_out_channel,
            cfg.model.densecls.layer1.conv1d_kernel_size
        )
        self.conv2 = torch.nn.Conv1d(
            cfg.model.densecls.layer2.conv1d_in_channel,
            cfg.model.densecls.layer2.conv1d_out_channel,
            cfg.model.densecls.layer2.conv1d_kernel_size
        )
        self.conv3 = torch.nn.Conv1d(
            cfg.model.densecls.layer3.conv1d_in_channel,
            cfg.model.densecls.layer3.conv1d_out_channel,
            cfg.model.densecls.layer3.conv1d_kernel_size
        )
        self.conv4 = torch.nn.Conv1d(
            cfg.model.densecls.layer4.conv1d_in_channel,
            self.num_seg_classes,
            cfg.model.densecls.layer3.conv1d_kernel_size
        )
        self.bn1 = nn.BatchNorm1d(cfg.model.densecls.layer1.conv1d_out_channel)
        self.bn2 = nn.BatchNorm1d(cfg.model.densecls.layer2.conv1d_out_channel)
        self.bn3 = nn.BatchNorm1d(cfg.model.densecls.layer3.conv1d_out_channel)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.num_seg_classes), dim=-1)
        x = x.view(batchsize, n_pts, self.num_seg_classes)
        return x, trans, trans_feat
    

class PointNetfeat(nn.Module):
    def __init__(self, cfg: DictConfig, global_feat = True):
        super().__init__()
        self.stn = STN3d(cfg)
        self.conv1 = torch.nn.Conv1d(
            cfg.model.feat.layer1.conv1d_in_channel,
            cfg.model.feat.layer1.conv1d_out_channel,
            cfg.model.feat.layer1.conv1d_kernel_size
        )
        self.conv2 = torch.nn.Conv1d(
            cfg.model.feat.layer2.conv1d_in_channel,
            cfg.model.feat.layer2.conv1d_out_channel,
            cfg.model.feat.layer2.conv1d_kernel_size
        )
        self.conv3 = torch.nn.Conv1d(
            cfg.model.feat.layer3.conv1d_in_channel,
            cfg.model.feat.layer3.conv1d_out_channel,
            cfg.model.feat.layer3.conv1d_kernel_size
        )
        self.bn1 = nn.BatchNorm1d(cfg.model.feat.layer1.conv1d_out_channel)
        self.bn2 = nn.BatchNorm1d(cfg.model.feat.layer2.conv1d_out_channel)
        self.bn3 = nn.BatchNorm1d(cfg.model.feat.layer3.conv1d_out_channel)
        self.global_feat = global_feat
        self.feature_transform = cfg.model.feat.feature_transform
        if self.feature_transform:
            self.fstn = STNkd(cfg)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        

class STN3d(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            cfg.model.stn.layer1.conv1d_in_channel,
            cfg.model.stn.layer1.conv1d_out_channel,
            cfg.model.stn.layer1.conv1d_kernel_size
        )

        self.conv2 = torch.nn.Conv1d(
            cfg.model.stn.layer2.conv1d_in_channel,
            cfg.model.stn.layer2.conv1d_out_channel,
            cfg.model.stn.layer2.conv1d_kernel_size
        )
    
        self.conv3 = torch.nn.Conv1d(
            cfg.model.stn.layer3.conv1d_in_channel,
            cfg.model.stn.layer3.conv1d_out_channel,
            cfg.model.stn.layer3.conv1d_kernel_size
        )
    
        self.fc1 = nn.Linear(cfg.model.stn.layer4.fc1_in_features, cfg.model.stn.layer4.fc1_out_features)
        self.fc2 = nn.Linear(cfg.model.stn.layer5.fc2_in_features, cfg.model.stn.layer5.fc2_out_features)
        self.fc3 = nn.Linear(cfg.model.stn.layer6.fc3_in_features, cfg.model.stn.layer6.fc3_out_features)
    
        self.bn1 = nn.BatchNorm1d(cfg.model.stn.layer1.conv1d_out_channel)
        self.bn2 = nn.BatchNorm1d(cfg.model.stn.layer2.conv1d_out_channel)
        self.bn3 = nn.BatchNorm1d(cfg.model.stn.layer3.conv1d_out_channel)
        self.bn4 = nn.BatchNorm1d(cfg.model.stn.layer4.fc1_out_features)
        self.bn5 = nn.BatchNorm1d(cfg.model.stn.layer5.fc2_out_features)
        
        relu = nn.ReLU()


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,.0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    

class STNkd(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.k = cfg.model.stn.k
        
        self.conv1 = torch.nn.Conv1d(
            self.k,
            cfg.model.stn.layer1.conv1d_out_channel,
            cfg.model.stn.layer1.conv1d_kernel_size
        )

        self.conv2 = torch.nn.Conv1d(
            cfg.model.stn.layer2.conv1d_in_channel,
            cfg.model.stn.layer2.conv1d_out_channel,
            cfg.model.stn.layer2.conv1d_kernel_size
        )
    
        self.conv3 = torch.nn.Conv1d(
            cfg.model.stn.layer3.conv1d_in_channel,
            cfg.model.stn.layer3.conv1d_out_channel,
            cfg.model.stn.layer3.conv1d_kernel_size
        )
    
        self.fc1 = nn.Linear(cfg.model.stn.layer4.fc1_in_features, cfg.model.stn.layer4.fc1_out_features)
        self.fc2 = nn.Linear(cfg.model.stn.layer5.fc2_in_features, cfg.model.stn.layer5.fc2_out_features)
        self.fc3 = nn.Linear(cfg.model.stn.layer6.fc3_in_features, self.k*self.k)
    
        self.bn1 = nn.BatchNorm1d(cfg.model.stn.layer1.conv1d_out_channel)
        self.bn2 = nn.BatchNorm1d(cfg.model.stn.layer2.conv1d_out_channel)
        self.bn3 = nn.BatchNorm1d(cfg.model.stn.layer3.conv1d_out_channel)
        self.bn4 = nn.BatchNorm1d(cfg.model.stn.layer4.fc1_out_features)
        self.bn5 = nn.BatchNorm1d(cfg.model.stn.layer5.fc2_out_features)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
