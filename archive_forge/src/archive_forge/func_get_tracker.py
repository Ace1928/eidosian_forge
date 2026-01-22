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
def get_tracker(self, name: str, unwrap: bool=False):
    """
        Returns a `tracker` from `self.trackers` based on `name` on the main process only.

        Args:
            name (`str`):
                The name of a tracker, corresponding to the `.name` property.
            unwrap (`bool`):
                Whether to return the internal tracking mechanism or to return the wrapped tracker instead
                (recommended).

        Returns:
            `GeneralTracker`: The tracker corresponding to `name` if it exists.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator(log_with="tensorboard")
        >>> accelerator.init_trackers("my_project")
        >>> tensorboard_tracker = accelerator.get_tracker("tensorboard")
        ```
        """
    if len(self.trackers) > 0:
        for tracker in self.trackers:
            if tracker.name == name:
                return tracker.tracker if unwrap else tracker
        raise ValueError(f'{name} is not an available tracker stored inside the `Accelerator`.')
    return GeneralTracker(_blank=True)