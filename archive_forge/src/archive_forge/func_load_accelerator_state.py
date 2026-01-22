import random
from pathlib import Path
from typing import List
import numpy as np
import torch
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from .utils import (
from .logging import get_logger
from .state import PartialState
def load_accelerator_state(input_dir, models, optimizers, schedulers, dataloaders, process_index, scaler=None, map_location=None, **load_model_func_kwargs):
    """
    Loads states of the models, optimizers, scaler, and RNG generators from a given directory.

    Args:
        input_dir (`str` or `os.PathLike`):
            The name of the folder to load all relevant weights and states.
        models (`List[torch.nn.Module]`):
            A list of model instances
        optimizers (`List[torch.optim.Optimizer]`):
            A list of optimizer instances
        schedulers (`List[torch.optim.lr_scheduler._LRScheduler]`):
            A list of learning rate schedulers
        process_index (`int`):
            The current process index in the Accelerator state
        scaler (`torch.cuda.amp.GradScaler`, *optional*):
            An optional *GradScaler* instance to load
        map_location (`str`, *optional*):
            What device to load the optimizer state onto. Should be one of either "cpu" or "on_device".
        load_model_func_kwargs (`dict`, *optional*):
            Additional arguments that can be passed to the model's `load_state_dict` method.
    """
    if map_location not in [None, 'cpu', 'on_device']:
        raise TypeError("Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`")
    if map_location is None:
        map_location = 'cpu'
    elif map_location == 'on_device':
        map_location = PartialState().device
    input_dir = Path(input_dir)
    for i, model in enumerate(models):
        ending = f'_{i}' if i > 0 else ''
        input_model_file = input_dir.joinpath(f'{SAFE_MODEL_NAME}{ending}.safetensors')
        if input_model_file.exists():
            state_dict = load_file(input_model_file, device=str(map_location))
        else:
            input_model_file = input_dir.joinpath(f'{MODEL_NAME}{ending}.bin')
            state_dict = torch.load(input_model_file, map_location=map_location)
        models[i].load_state_dict(state_dict, **load_model_func_kwargs)
    logger.info('All model weights loaded successfully')
    for i, opt in enumerate(optimizers):
        optimizer_name = f'{OPTIMIZER_NAME}.bin' if i == 0 else f'{OPTIMIZER_NAME}_{i}.bin'
        input_optimizer_file = input_dir.joinpath(optimizer_name)
        optimizer_state = torch.load(input_optimizer_file, map_location=map_location)
        optimizers[i].load_state_dict(optimizer_state)
    logger.info('All optimizer states loaded successfully')
    for i, scheduler in enumerate(schedulers):
        scheduler_name = f'{SCHEDULER_NAME}.bin' if i == 0 else f'{SCHEDULER_NAME}_{i}.bin'
        input_scheduler_file = input_dir.joinpath(scheduler_name)
        scheduler.load_state_dict(torch.load(input_scheduler_file))
    logger.info('All scheduler states loaded successfully')
    for i, dataloader in enumerate(dataloaders):
        sampler_name = f'{SAMPLER_NAME}.bin' if i == 0 else f'{SAMPLER_NAME}_{i}.bin'
        input_sampler_file = input_dir.joinpath(sampler_name)
        from .data_loader import IterableDatasetShard, SeedableRandomSampler
        if isinstance(dataloader.dataset, IterableDatasetShard):
            sampler = dataloader.sampler.sampler
            if isinstance(sampler, SeedableRandomSampler):
                dataloader.sampler.sampler = torch.load(input_sampler_file)
    logger.info('All dataloader sampler states loaded successfully')
    if scaler is not None:
        input_scaler_file = input_dir.joinpath(SCALER_NAME)
        scaler.load_state_dict(torch.load(input_scaler_file))
        logger.info('GradScaler state loaded successfully')
    try:
        states = torch.load(input_dir.joinpath(f'{RNG_STATE_NAME}_{process_index}.pkl'))
        random.setstate(states['random_state'])
        np.random.set_state(states['numpy_random_seed'])
        torch.set_rng_state(states['torch_manual_seed'])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states['torch_xpu_manual_seed'])
        else:
            torch.cuda.set_rng_state_all(states['torch_cuda_manual_seed'])
        if is_torch_xla_available():
            xm.set_rng_state(states['xm_seed'])
        logger.info('All random states loaded successfully')
    except Exception:
        logger.info('Could not load random states')