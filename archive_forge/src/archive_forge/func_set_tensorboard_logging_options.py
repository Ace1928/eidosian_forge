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
def set_tensorboard_logging_options(self):
    from megatron.arguments import _add_logging_args
    parser = argparse.ArgumentParser()
    parser = _add_logging_args(parser)
    logging_args = parser.parse_known_args()
    self.dataset_args = vars(logging_args[0])
    for key, value in self.dataset_args.items():
        if key.startswith('log_'):
            self.megatron_lm_default_args[key] = True
        elif key.startswith('no_log_'):
            self.megatron_lm_default_args[key.replace('no_', '')] = True