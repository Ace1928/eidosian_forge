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
def remove_node_cached_tensors(self):
    for t in self.cached_tensor_outputs:
        if t is not None:
            torch._C._remove_cached_tensor(t)
    self.cached_tensor_outputs.clear()
    for i, unaliased in enumerate(self.unaliased_in_all_paths):
        if unaliased:
            n = self.outputs_weakrefs[i]
            assert n is not None
            n.remove_extra_reference()