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
def data_ptrs_dead_since_invocation(self) -> List[int]:
    """
        Since this node was invoked, return data ptrs of all tensor outputs that have died
        in the current executing tree path.
        """
    curr_liveness = self._get_liveness(self.path_weakrefs)
    _get_different_indices = self._get_different_indices(self.recorded_liveness_after_graph, curr_liveness)
    path = list(self._path_from_root)
    ptrs_to_deallocate = []
    for depth, output_index in _get_different_indices:
        ptrs_to_deallocate.append(path[depth].outputs_metadata[output_index]['data_ptr'])
    return ptrs_to_deallocate