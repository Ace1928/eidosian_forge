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
def set_mixed_precision(self, mixed_precision):
    if mixed_precision == 'fp16':
        self.megatron_lm_default_args['fp16'] = True
    elif mixed_precision == 'bf16':
        self.megatron_lm_default_args['bf16'] = True
        self.DDP_impl = 'local'
        self.megatron_lm_default_args['DDP_impl'] = self.DDP_impl