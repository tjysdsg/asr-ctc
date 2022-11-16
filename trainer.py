import json
import os
import os.path
import math
import time
import logging
import configargparse

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.asr_model import ASRModel
from models.scheduler import WarmupLR
from loader import create_loader
from utils import to_device


class Trainer:
    def __init__(self, params: configargparse.Namespace):
        """Initializes the Trainer with the training args provided in train.py"""

        logging.basicConfig(
            filename=os.path.join(params.log_dir, "train.log"),
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )

        self.params = params
        self.nepochs = params.nepochs
        self.ngpu = params.ngpu
        with open(params.train_json, "rb") as f:
            train_json = json.load(f)["utts"]
        with open(params.valid_json, "rb") as f:
            valid_json = json.load(f)["utts"]

        self.train_dataset, self.train_sampler, _ = create_loader(
            train_json, params, is_train=True
        )
        self.valid_dataset, self.valid_sampler, _ = create_loader(
            valid_json, params, is_train=False
        )

        ## Build Model
        self.model = ASRModel(params)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        if self.ngpu > 1:
            self.model = DataParallel(self.model)

        logging.info(str(self.model))

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        params.tparams = total_params
        logging.info(
            "Built a model with {:2.2f}M Params".format(float(total_params) / 1000000)
        )

        ## Write out model config
        with open(os.path.join(params.expdir, "model.json"), "wb") as f:
            f.write(json.dumps(vars(params), indent=4, sort_keys=True).encode("utf_8"))

        ## Optimizer
        self.opt = Adam(
            self.model.parameters(), lr=params.lr, weight_decay=params.wdecay
        )
        self.scheduler = WarmupLR(self.opt, warmup_steps=params.warmup_steps)

        ## Initialize Stats for Logging
        self.train_stats = {}
        self.val_stats = {"best_loss": 1e9, "best_epoch": -1}
        self.writer = SummaryWriter(self.params.tb_dir)  # for tensorboard

        ## Resume/Load Model
        if params.resume != "":
            self.resume_training(params.resume)
        else:
            self.epoch = 0
        self.start_time = time.time()

    def train(self):
        """Performs ASR Training using the provided configuration.
        This is the main training wrapper that trains and evaluates the model across epochs
        """
        while self.epoch < self.nepochs:
            self.reset_stats()
            start_time = time.time()

            logging.info(f"Start to train epoch {self.epoch}")
            self.train_epoch()

            logging.info(f"Start to validate epoch {self.epoch}")
            self.validate_epoch()
            end_time = time.time()

            ## Log Tensorboard and logfile
            log_str = (
                f"Epoch {self.epoch:02d}, lr={self.opt.param_groups[0]['lr']} | Train: loss={self.train_stats['loss']:.4f}, wer={self.train_stats['wer']:.4f}"
                f" | Val: loss={self.val_stats['loss']:.4f}, wer={self.val_stats['wer']:.4f} | "
                f"Time: this epoch {end_time - start_time:.2f}s, elapsed {end_time - self.start_time:.2f}s"
            )
            logging.info(log_str)
            self.log_epoch()

            self.save_model()
            self.epoch += 1

    def train_epoch(self):
        """ "Contains the training loop across all training data to update the model in an epoch"""
        self.model.train()

        for i, (feats, feat_lens, target, target_lens, train_keys) in enumerate(
                self.train_sampler
        ):
            feats, feat_lens, target, target_lens = to_device(
                (feats, feat_lens, target, target_lens),
                next(self.model.parameters()).device,
            )

            loss, wer = self.model(feats, feat_lens, target, target_lens)
            loss /= self.params.accum_grad
            loss.backward()

            if (i + 1) % self.params.log_interval == 0:
                logging.info(
                    f"[Epoch {self.epoch}, Batch={i}] Train: loss={loss.item():.4f}, wer={wer:.4f}, lr={self.opt.param_groups[0]['lr']}"
                )

            if (i + 1) % self.params.accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                if math.isnan(grad_norm):
                    logging.info("[Warning] Grad norm is nan. Do not update model.")
                else:
                    self.opt.step()
                    self.scheduler.step()

                self.opt.zero_grad()

            self.train_stats["nbatches"] += 1
            self.train_stats["loss"] += loss.item()
            self.train_stats["wer"] += wer

        self.train_stats["loss"] /= self.train_stats["nbatches"]
        self.train_stats["wer"] /= self.train_stats["nbatches"]

    def validate_epoch(self):
        """ "Contains the validation loop across all validation data in an epoch"""
        self.model.eval()

        with torch.no_grad():
            for i, (feats, feat_lens, target, target_lens, valid_keys) in enumerate(
                    self.valid_sampler
            ):
                feats, feat_lens, target, target_lens = to_device(
                    (feats, feat_lens, target, target_lens),
                    next(self.model.parameters()).device,
                )

                loss, wer = self.model(feats, feat_lens, target, target_lens)

                self.val_stats["nbatches"] += 1
                self.val_stats["loss"] += loss.item()
                self.val_stats["wer"] += wer

            self.val_stats["loss"] /= self.val_stats["nbatches"]
            self.val_stats["wer"] /= self.val_stats["nbatches"]

    def resume_training(self, path: str):
        """
        Utility function to load a previous model and optimizer checkpoint, and set the starting epoch for resuming training
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.val_stats["best_epoch"] = checkpoint["epoch"]
        self.val_stats["best_loss"] = checkpoint["loss"]

    def reset_stats(self):
        """
        Utility function to reset training and validation statistics at the start of each epoch
        """
        self.train_stats["nbatches"] = 0
        self.train_stats["loss"] = 0
        self.train_stats["wer"] = 0

        self.val_stats["nbatches"] = 0
        self.val_stats["loss"] = 0
        self.val_stats["wer"] = 0

    def save_model(self):
        """Save the model snapshot after every epoch of training."""
        if self.val_stats["loss"] < self.val_stats["best_loss"]:
            # TODO: keep nbest models and average them
            # old_ckpt = os.path.join(
            #     self.params.model_dir, f'epoch{self.val_stats["best_epoch"]}.pth'
            # )
            # if os.path.exists(old_ckpt):
            #     os.remove(old_ckpt)
            self.val_stats["best_epoch"] = self.epoch
            self.val_stats["best_loss"] = self.val_stats["loss"]

            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "loss": self.val_stats["loss"],
                    "wer": self.val_stats["wer"],
                },
                os.path.join(self.params.model_dir, f"epoch{self.epoch}.pth"),
            )
            logging.info(f"[info] Save model after epoch {self.epoch}\n")

    def log_epoch(self):
        """Write stats from the Training and Validation Statistics Dictionaries onto Tensorboard at the end of each epoch"""
        self.writer.add_scalar("training/loss", self.train_stats["loss"], self.epoch)
        self.writer.add_scalar("training/wer", self.train_stats["wer"], self.epoch)
        self.writer.add_scalar("validation/loss", self.val_stats["loss"], self.epoch)
        self.writer.add_scalar("validation/wer", self.val_stats["wer"], self.epoch)
