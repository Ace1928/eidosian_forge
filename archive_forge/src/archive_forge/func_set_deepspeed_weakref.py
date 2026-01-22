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
def set_deepspeed_weakref(self):
    from .imports import is_transformers_available
    if self.zero3_init_flag:
        if not is_transformers_available():
            raise Exception('When `zero3_init_flag` is set, it requires Transformers to be installed. Please run `pip install transformers`.')
        ds_config = copy.deepcopy(self.deepspeed_config)
        if 'gradient_accumulation_steps' not in ds_config or ds_config['gradient_accumulation_steps'] == 'auto':
            ds_config['gradient_accumulation_steps'] = 1
        if 'train_micro_batch_size_per_gpu' not in ds_config or ds_config['train_micro_batch_size_per_gpu'] == 'auto':
            ds_config['train_micro_batch_size_per_gpu'] = 1
        if ds_config.get('train_batch_size', None) == 'auto':
            del ds_config['train_batch_size']
        if compare_versions('transformers', '<', '4.33'):
            from transformers.deepspeed import HfDeepSpeedConfig
        else:
            from transformers.integrations import HfDeepSpeedConfig
        self.dschf = HfDeepSpeedConfig(ds_config)