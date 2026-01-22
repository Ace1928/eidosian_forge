import functools
import itertools
import logging
import os
import queue
import threading
import warnings
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from . import (
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from . import _utils
@multiprocessing_context.setter
def multiprocessing_context(self, multiprocessing_context):
    if multiprocessing_context is not None:
        if self.num_workers > 0:
            if isinstance(multiprocessing_context, str):
                valid_start_methods = multiprocessing.get_all_start_methods()
                if multiprocessing_context not in valid_start_methods:
                    raise ValueError(f'multiprocessing_context option should specify a valid start method in {valid_start_methods!r}, but got multiprocessing_context={multiprocessing_context!r}')
                multiprocessing_context = multiprocessing.get_context(multiprocessing_context)
            if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                raise TypeError(f'multiprocessing_context option should be a valid context object or a string specifying the start method, but got multiprocessing_context={multiprocessing_context}')
        else:
            raise ValueError(f'multiprocessing_context can only be used with multi-process loading (num_workers > 0), but got num_workers={self.num_workers}')
    self.__multiprocessing_context = multiprocessing_context