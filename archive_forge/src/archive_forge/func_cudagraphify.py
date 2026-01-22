important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
def cudagraphify(model, inputs, static_input_idxs=(), *, device_index: int, is_backward: bool, is_inference: bool, stack_traces: Optional[StackTraces]=None, constants: Tuple[torch.Tensor, ...]=()):
    manager = get_container(device_index).get_tree_manager()
    assert not (is_backward and is_inference)
    mode = CompilationMode.BACKWARD if is_backward else CompilationMode.INFERENCE if is_inference else CompilationMode.FORWARD
    return manager.add_function(model, inputs, static_input_idxs, stack_traces, mode, constants)