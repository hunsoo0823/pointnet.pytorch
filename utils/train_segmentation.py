from typing import Optional
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
import torch
from datetime import datetime
import os
import sys
import yaml

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from pointnet.model import PointNetDenseCls
from data_utils import dataset_split
from config_utils import register_config
from config_utils import get_loggers
from config_utils import get_callbacks
from pointnet.dataset import ShapeNetDataset


def load_yaml(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

output_dir = os.path.join(parent_dir, "config_yaml")
data_root = os.path.join(parent_dir, "data")


model_cfg_file = os.path.join(output_dir, "model_pointnet.yaml")
optimizer_cfg_file = os.path.join(output_dir, "optimizer.yaml")
data_cfg_file = os.path.join(output_dir, "data_shapenet.yaml")

model_cfg = load_yaml(model_cfg_file)
optimizer_cfg = load_yaml(optimizer_cfg_file)
data_cfg = load_yaml(data_cfg_file)
data_cfg["data_root"] = data_root


_merged_cfg_presets = {
    "PointNet": {
        "opt": optimizer_cfg,
        "data": data_cfg,
        "model": model_cfg,
    }
}

# clear config instance first
hydra.core.global_hydra.GlobalHydra.instance().clear()

# register preset configs
register_config(_merged_cfg_presets)

# initialize & mae config
## select mode here ##
# .................. #
hydra.initialize(config_path=None, version_base="1.1")
cfg = hydra.compose("PointNet")

# override some cfg
run_name = f"{datetime.now().isoformat(timespec='seconds')}-{cfg.model.name}-{cfg.data.name}"

# Define other train configs & log_configs
# Merge configs into one & register it to Hydra.

project_root_dir = os.path.join(parent_dir, "runs", "pointnet-runs")


save_dir = os.path.join(project_root_dir, run_name)
run_root_dir = os.path.join(project_root_dir, run_name)

train_cfg_file = os.path.join(output_dir, "train.yaml")
log_cfg_file = os.path.join(output_dir, "log.yaml")

train_cfg = load_yaml(train_cfg_file)
train_cfg["run_root_dir"] = run_root_dir
log_cfg = load_yaml(log_cfg_file)
log_cfg["loggers"]["WandbLogger"]["project"] = "pointNet_seg"
log_cfg["loggers"]["WandbLogger"]["name"] = run_name
log_cfg["loggers"]["WandbLogger"]["save_dir"] = run_root_dir
log_cfg["callbacks"]["ModelCheckpoint"]["dirpath"] = os.path.join(run_root_dir, "weights")


OmegaConf.set_struct(cfg, False)
cfg.train = train_cfg
cfg.log = log_cfg

# lock config
OmegaConf.set_struct(cfg, True)

data_root = cfg.data.data_root

dataset = ShapeNetDataset(
    root=data_root, 
    classification=False,
    class_choice=cfg.data.class_choice
)

test_dataset = ShapeNetDataset(
    root=data_root,
    classification=False,
    split='test',
    class_choice=cfg.data.class_choice,
    data_augmentation=False
)

num_seg_classes = dataset.num_seg_classes

dataset = dataset_split(dataset, split=cfg.train.train_val_split)
train_dataset = dataset["train"]
val_dataset = dataset["val"]

train_batch_size = cfg.train.train_batch_size
val_batch_size = cfg.train.val_batch_size
test_batch_size = cfg.train.test_batch_size

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=0
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0
)

def get_pl_model(cfg: DictConfig, checkpoint_path: Optional[str] = None):

    if cfg.model.name == "PointNet":
        model = PointNetDenseCls(cfg, num_seg_classes=num_seg_classes)
    else:
        raise NotImplementedError()

    if checkpoint_path is not None:
        model = model.load_from_checkpoint(checkpoint_path)

    return model

model = get_pl_model(cfg)

logger = get_loggers(cfg)
callbacks = get_callbacks(cfg)

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=cfg.train.run_root_dir,
    num_sanity_val_steps=2,
    **cfg.train.trainer_kwargs
)

trainer.fit(model, train_dataloader, val_dataloader)