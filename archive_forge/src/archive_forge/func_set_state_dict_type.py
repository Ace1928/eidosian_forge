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
def set_state_dict_type(self, state_dict_type_policy):
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
    self.state_dict_type = StateDictType(FSDP_STATE_DICT_TYPE.index(state_dict_type_policy) + 1)
    if self.state_dict_type == StateDictType.FULL_STATE_DICT:
        if self.state_dict_config is None:
            self.state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if self.optim_state_dict_config is None:
            self.optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)