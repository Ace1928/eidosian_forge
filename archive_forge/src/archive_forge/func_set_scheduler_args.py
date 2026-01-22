import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
def set_scheduler_args(self, scheduler):
    if self.train_iters is None:
        self.train_iters = scheduler.total_num_steps // self.megatron_lm_default_args['data_parallel_size']
        if self.train_samples is not None:
            self.train_samples = None
            warnings.warn('Ignoring `train_samples` as `train_iters` based on scheduler is being used for training.')
    if self.lr_warmup_iters is None:
        self.lr_warmup_iters = scheduler.warmup_num_steps // self.megatron_lm_default_args['data_parallel_size']
        if self.lr_warmup_samples is not None:
            warnings.warn('Ignoring `lr_warmup_samples` as `lr_warmup_iters` based on scheduler is being used for training.')
        self.lr_warmup_samples = 0
    self.megatron_lm_default_args['train_iters'] = self.train_iters
    self.megatron_lm_default_args['lr_warmup_iters'] = self.lr_warmup_iters
    self.megatron_lm_default_args['train_samples'] = self.train_samples
    self.megatron_lm_default_args['lr_warmup_samples'] = self.lr_warmup_samples
    self.megatron_lm_default_args['lr_decay_iters'] = self.lr_decay_iters
    self.megatron_lm_default_args['lr_decay_samples'] = self.lr_decay_samples
    self.megatron_lm_default_args['lr_warmup_fraction'] = self.lr_warmup_fraction
    self.megatron_lm_default_args['lr_decay_style'] = self.lr_decay_style
    self.megatron_lm_default_args['weight_decay_incr_style'] = self.weight_decay_incr_style
    self.megatron_lm_default_args['start_weight_decay'] = self.start_weight_decay
    self.megatron_lm_default_args['end_weight_decay'] = self.end_weight_decay
    self.megatron_lm_default_args['min_lr'] = self.min_lr