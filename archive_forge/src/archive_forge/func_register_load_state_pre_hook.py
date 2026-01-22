from __future__ import annotations
import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union
import torch
import torch.utils.hooks as hooks
from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module
from torch.distributed.algorithms.join import Join
def register_load_state_pre_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
    """
        Registers a pre hook to be run before [`load_checkpoint`] is called in [`Accelerator.load_state`].

        Args:
            hook (`Callable`):
                A function to be called in [`Accelerator.load_state`] before `load_checkpoint`.

        The hook should have the following signature:

        `hook(models: list[torch.nn.Module], input_dir: str) -> None`

        The `models` argument are the models as saved in the accelerator state under `accelerator._models`, and the
        `input_dir` argument is the `input_dir` argument passed to [`Accelerator.load_state`].

        <Tip>

        Should only be used in conjunction with [`Accelerator.register_save_state_pre_hook`]. Can be useful to load
        configurations in addition to model weights. Can also be used to overwrite model loading with a customized
        method. In this case, make sure to remove already loaded models from the models list.

        </Tip>

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling
            `handle.remove()`
        """
    handle = hooks.RemovableHandle(self._load_model_state_pre_hook)
    self._load_model_state_pre_hook[handle.id] = hook
    return handle