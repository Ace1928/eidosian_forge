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
def to_kwargs(self):
    """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
    from .other import clear_environment
    with clear_environment():
        default_dict = self.__class__().to_dict()
    this_dict = self.to_dict()
    return {k: v for k, v in this_dict.items() if default_dict[k] != v}