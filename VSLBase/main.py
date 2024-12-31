"""Main script to train/test models for Ego4D NLQ dataset using VSLBase."""

import argparse
import os
import numpy as np
import options
import torch
import torch.nn as nn
import submitit
from torch.utils.tensorboard.writer import SummaryWriter
from model.VSLBase import build_optimizer_and_scheduler, VSLBase
from tqdm import tqdm
from utils.data_gen import gen_or_load_dataset
from utils.data_loader import get_test_loader, get_train_loader
from utils.data_util import load_json, load_video_features, save_json
from utils.runner_utils import (
    convert_length_to_mask,
    eval_test,
    filter_checkpoints,
    get_last_checkpoint,
    set_th_config,
)

def main(configs, parser):
    print(f"Running with {configs}", flush=True)

    # Set PyTorch configurations
    set_th_config(configs.seed)

    # Load or generate dataset
    dataset = gen_or_load_dataset(configs)
    configs.char_size = dataset.get("n_chars", -1)
    configs.word_size = dataset.get("n_words", -1)

    # Load video features
    visual_features = load_video_features(
        os.path.join("data", "features", configs.task, configs.fv), configs.max_pos_len
    )
    if configs.video_agnostic:
        visual_features = {
            key: np.random.rand(*val.shape) for key, val in visual_features.items()
        }

    # Prepare data loaders
    train_loader = get_train_loader(
        dataset=dataset["train_set"], video_features=visual_features, configs=configs
    )
    val_loader = (
        None
        if dataset["val_set"] is None
        else get_test_loader(dataset["val_set"], visual_features, configs)
    )
    test_loader = get_test_loader(
        dataset=dataset["test_set"], video_features=visual_features, configs=configs
    )

    # Set device
    cuda_str = "cuda" if configs.gpu_idx is None else f"cuda:{configs.gpu_idx}"
    device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    # Set model directory
    model_dir = os.path.join(
        configs.model_dir,
        f"{configs.model_name}_{configs.task}_{configs.fv}_{configs.max_pos_len}_{configs.predictor}",
    )
    if configs.suffix:
        model_dir += f"_{configs.suffix}"
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer
    writer = None
    if configs.log_to_tensorboard:
        log_dir = os.path.join(configs.tb_log_dir, configs.log_to_tensorboard)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    # Training and testing logic
    if configs.mode.lower() == "train":
        # Save configurations
        save_json(vars(configs), os.path.join(model_dir, "configs.json"), sort_keys=True, save_pretty=True)

        # Build model
        model = VSLBase(configs=configs).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)

        # Training loop
        best_metric = -1.0
        global_step = 0
        for epoch in range(configs.epochs):
            model.train()
            for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.epochs}"):
                global_step += 1
                (
                    _,
                    vfeats,
                    vfeat_lens,
                    word_ids,
                    char_ids,
                    s_labels,
                    e_labels,
                    h_labels,
                ) = data

                # Move data to device
                vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
                if configs.predictor == "bert":
                    word_ids = {key: val.to(device) for key, val in word_ids.items()}
                    query_mask = (word_ids["input_ids"] != 0).float().to(device)
                else:
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    query_mask = (word_ids != 0).float().to(device)
                video_mask = convert_length_to_mask(vfeat_lens).to(device)

                # Forward pass
                h_score, start_logits, end_logits = model(
                    word_ids, char_ids, vfeats, video_mask, query_mask
                )

                # Compute losses
                highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask)
                loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)
                total_loss = loc_loss + configs.highlight_lambda * highlight_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)
                optimizer.step()
                scheduler.step()

                # Logging
                if writer and global_step % configs.tb_log_freq == 0:
                    writer.add_scalar("Loss/Total", total_loss.item(), global_step)

        # Save final model
        torch.save(model.state_dict(), os.path.join(model_dir, f"{configs.model_name}_final.t7"))

    elif configs.mode.lower() == "test":
        raise NotImplementedError("Testing mode has not been fully implemented.")

if __name__ == "__main__":
    configs, parser = options.read_command_line()
    main(configs, parser)
