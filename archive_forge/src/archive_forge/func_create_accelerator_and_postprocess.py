import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def create_accelerator_and_postprocess(self):
    grad_acc_kwargs = {'num_steps': self.args.gradient_accumulation_steps}
    grad_acc_kwargs['sync_with_dataloader'] = False
    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
    accelerator_kwargs = {}
    if self.args.accelerator_config is not None:
        accelerator_kwargs = self.args.accelerator_config
        if isinstance(accelerator_kwargs, AcceleratorConfig):
            accelerator_kwargs = accelerator_kwargs.to_dict()
        elif isinstance(accelerator_kwargs, dict):
            accelerator_kwargs = AcceleratorConfig(**accelerator_kwargs).to_dict()
    self.accelerator = Accelerator(deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, **accelerator_kwargs)
    self.gather_function = self.accelerator.gather_for_metrics
    self.is_deepspeed_enabled = getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
    self.is_fsdp_enabled = getattr(self.accelerator.state, 'fsdp_plugin', None) is not None
    if self.is_fsdp_enabled:
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get('limit_all_gathers', fsdp_plugin.limit_all_gathers)
        if is_accelerate_available('0.23.0'):
            fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get('activation_checkpointing', fsdp_plugin.activation_checkpointing)
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic when using FSDP.")
    if self.is_deepspeed_enabled and getattr(self.args, 'hf_deepspeed_config', None) is None:
        self.propagate_args_to_deepspeed()
    if self.args.save_only_model and (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.load_best_model_at_end:
        wrapper = 'DeepSpeed' if self.is_deepspeed_enabled else 'FSDP'
        raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")
    if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.auto_find_batch_size:
        wrapper = 'DeepSpeed' if self.is_deepspeed_enabled else 'FSDP'
        raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")