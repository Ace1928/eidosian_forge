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
@contextmanager
def no_sync(self, model):
    """
        A context manager to disable gradient synchronizations across DDP processes by calling
        `torch.nn.parallel.DistributedDataParallel.no_sync`.

        If `model` is not in DDP, this context manager does nothing

        Args:
            model (`torch.nn.Module`):
                PyTorch Module that was prepared with `Accelerator.prepare`

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
        >>> input_a = next(iter(dataloader))
        >>> input_b = next(iter(dataloader))

        >>> with accelerator.no_sync():
        ...     outputs = model(input_a)
        ...     loss = loss_func(outputs)
        ...     accelerator.backward(loss)
        ...     # No synchronization across processes, only accumulate gradients
        >>> outputs = model(input_b)
        >>> accelerator.backward(loss)
        >>> # Synchronization across all processes
        >>> optimizer.step()
        >>> optimizer.zero_grad()
        ```
        """
    context = contextlib.nullcontext
    if self.use_distributed:
        context = getattr(model, 'no_sync', context)
    with context():
        yield