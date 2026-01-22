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
def set_trigger(self):
    """
        Sets the internal trigger tensor to 1 on the current process. A latter check should follow using this which
        will check across all processes.

        Note:
            Does not require `wait_for_everyone()`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> # Assume later in the training script
        >>> # `should_do_breakpoint` is a custom function to monitor when to break,
        >>> # e.g. when the loss is NaN
        >>> if should_do_breakpoint(loss):
        ...     accelerator.set_trigger()
        >>> # Assume later in the training script
        >>> if accelerator.check_breakpoint():
        ...     break
        ```
        """
    self.flag_tensor = torch.tensor(1, device=self.device)