import os
import time
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from itertools import chain
import numpy as np

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts,return_all=False):
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    if return_all:
        return avg_cost,cost.max(),cost.min(),torch.std(cost) / math.sqrt(len(cost)),cost
    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

def eval_loss(model, dataset,baseline, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    dataset=baseline.wrap_dataset(dataset)
    @torch.no_grad()
    def eval_model_bat(batch):
        x, bl_val = baseline.unwrap_batch(batch)
        x = move_to(x, opts.device)
        bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

        # Evaluate model, get costs and log probabilities
        cost, log_likelihood = model(x)

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
        return ((cost-bl_val)*log_likelihood).cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def train_batch(
        model,
        optimizer,
        baseline,
        batch,
        device,
        opts
):
    '''
    simple train batch
    '''
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, device)
    bl_val = move_to(bl_val,device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    loss=(cost - bl_val) * log_likelihood
    reinforce_loss = (loss).mean()

    loss = reinforce_loss + bl_loss
    
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    return cost.mean().item()



def train_epoch(model, optimizer, baseline, opts,train_dataset=None,epoch=1):
    '''
    simple train epoch
    '''
    training_dataset = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1,shuffle=True)
            
    model.train()
    set_decode_type(model, "sampling")
    bar=tqdm(training_dataloader, disable=opts.no_progress_bar)
    costs=[]
    for batch_id, batch in enumerate(bar):
        cost=train_batch(
            model,
            optimizer,
            baseline,
            batch,
        )
        bar.set_postfix(cost=cost)
        costs.append(cost)
    baseline.epoch_callback(model, epoch,dataset=train_dataset)
    return np.mean(costs)