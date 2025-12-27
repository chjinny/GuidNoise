import wandb
import yaml
import importlib
import torch
import os
import sys
from models.model import load_model
from diffusion import Diffusion
from tqdm import tqdm
import numpy as np
from dataset.dataset import Dataset, cycle_dataloader, num_to_groups


class Trainer():
    def __init__(self, config, wandb_enabled=True):
        self.config = config
        self.wandb_enabled = wandb_enabled

        model, model_config = load_model(**self.config["model"])
        self.diffusion = Diffusion(model.cuda(), **self.config["diffusion"]).cuda()

        self.optimizer = getattr(importlib.import_module(f"torch.optim"), self.config["optimizer"]["name"])(
            self.diffusion.parameters(),
            lr = self.config["optimizer"]["lr"],
            **self.config["optimizer"]["params"]
        )
        self.config["model"]["config"] = str(model_config)

        train_dataset = Dataset(**self.config["data"]["train"], max_iteration=self.config["max_iteration"])
        self.trainloader = cycle_dataloader(torch.utils.data.DataLoader(train_dataset, batch_size=self.config["data"]["train"]["batch_size"], shuffle=True))
        self.val_dataset = Dataset(**self.config["data"]["val"])
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset)
        print(f"train dataset length: {len(train_dataset)}", f"val dataset length: {len(self.val_dataset)}")

        self.step = 0
        self.init_step = 0
        self.num_val = 0
        self.metric_dict = {metric : getattr(importlib.import_module(f"metrics"),f"batch_{metric}") for metric in self.config["metrics"]}

        if self.config["checkpoint_path"] is not None:
            self.load(self.config["checkpoint_path"])

        if self.wandb_enabled:
            self.run = wandb.init(
                project="generator", 
                config=self.config,
            )
            log_unit_list = ["iteration_train", "iteration_val", "val"]
            for log_unit in log_unit_list:
                self.run.define_metric(
                    name=log_unit,
                    summary="mean",
                )
                for key in self.metric_dict:
                    self.run.define_metric(
                        name=f"{log_unit}_{key}",
                        step_metric=log_unit,
                    )
            self.run.define_metric(
                name=f"iteration_train_loss",
                step_metric=f"iteration_train",
            )
        else:
            self.run = None
    
    def train(self):
        loss_list = []
        self.diffusion.train()
        max_iteration = self.config["max_iteration"]
        with tqdm(initial = self.step, total = self.config["max_iteration"], ncols=100, unit_scale=True, unit="") as pbar:
            while self.step < max_iteration:
                data = next(self.trainloader)
                loss = self.diffusion(
                    noisy_image=data["lq"].cuda(),
                    clean_image=data["hq"].cuda(), 
                    refer_noisy=data["lq"].cuda(), 
                    refer_clean=data["hq"].cuda()
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.step%10 == 0:
                    if self.wandb_enabled:
                        self.run.log({f"iteration_train":self.step, f"iteration_train_loss":loss.item()})
                loss_list.append(loss.item())
                if len(loss_list) > 100:
                    loss_list = loss_list[-100:]
                if self.step != self.init_step and self.step % self.config["val_interval"] == 0:
                    self.save()
                    self.val()
                self.step+=1
                pbar.set_postfix_str(f"loss: {np.mean(loss_list):.4f}")
                pbar.update(1)
    
    @torch.inference_mode()
    def val(self):
        logs = {}
        pbar = tqdm(self.val_loader, ncols=100, desc=f"val")
        batches = num_to_groups(1, self.val_loader.batch_size)
        self.diffusion.eval()
        for i, data in enumerate(pbar):
            samples = torch.cat(list(map(lambda n: self.diffusion.sample(
                batch_size=n,
                clean_image=data["hq"].cuda(),
                refer_noisy=data["lq"].cuda(),
                refer_clean=data["hq"].cuda(),    
            ), batches)), dim = 0).clip(0,1)

            for metric in self.metric_dict:
                logs[metric] = logs.get(metric, [])
                logs[metric].append(list(self.metric_dict[metric](samples, data["lq"].cuda(), data["hq"].cuda())))
            if self.wandb_enabled:
                self.run.log({f"iteration_val": self.num_val * len(pbar) + i, **{f"iteration_val_{metric}": np.mean(logs[metric]) for metric in logs}})
            pbar.set_postfix(**{metric:f"{np.mean(logs[metric]):.3f}" for metric in logs})
        if self.wandb_enabled:
            self.run.log({f"val": self.num_val, **{f"val_{metric}": np.mean(logs[metric]) for metric in logs}})
        self.num_val += 1
        self.diffusion.train()

    def save(self, path_dir=None):
        if path_dir is None:
            if self.wandb_enabled:
                path_dir = self.run.dir
            else:
                path_dir = "."
        torch.save({
            "diffusion": self.diffusion.state_dict(),
        }, path_dir + "/model.pth")
        print(f"model saved at {path_dir}")

    def load(self, path_dir=None):
        if path_dir is None:
            path_dir = self.run.dir
        checkpoint = torch.load(path_dir + "/model.pth")
        self.diffusion.load_state_dict(checkpoint["diffusion"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sidd_val.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config, wandb_enabled=False)
    # trainer.train()
    trainer.val()