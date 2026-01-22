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
def try_end_curr_warmup(self, function_id: FunctionID):
    if self.can_start_new_generation():
        self.dealloc_current_path_weakrefs()
        self.current_node = None
        return
    if self.current_node.all_outputs_are_dead():
        self.current_node = None
        return
    self.check_warn_on_unable_to_start_executing(function_id)