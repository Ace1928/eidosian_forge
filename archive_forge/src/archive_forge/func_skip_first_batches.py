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
def skip_first_batches(self, dataloader, num_batches: int=0):
    """
        Creates a new `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.

        Args:
            dataloader (`torch.utils.data.DataLoader`): The data loader in which to skip batches.
            num_batches (`int`, *optional*, defaults to 0): The number of batches to skip

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
        >>> skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=2)
        >>> # for the first epoch only
        >>> for input, target in skipped_dataloader:
        ...     optimizer.zero_grad()
        ...     output = model(input)
        ...     loss = loss_func(output, target)
        ...     accelerator.backward(loss)
        ...     optimizer.step()

        >>> # subsequent epochs
        >>> for input, target in dataloader:
        ...     optimizer.zero_grad()
        ...     ...
        ```
        """
    return skip_first_batches(dataloader, num_batches=num_batches)