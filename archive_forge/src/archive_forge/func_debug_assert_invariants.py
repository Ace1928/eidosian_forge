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
def debug_assert_invariants(self, expected_liveness: List[List[bool]], newly_dead: List[PathOutputIndex]):
    if not config.triton.fast_path_cudagraph_asserts:
        return
    for i, node in enumerate(self._path_from_root):
        assert self.path_weakrefs[i] is node.outputs_weakrefs
    nodes = list(self._path_from_root)
    live_blocks = get_block_addrs(self.cuda_graphs_pool)
    live_storage_data_ptrs = set()
    live_storage_weak_ptrs = set()
    for depth, outputs_liveness in enumerate(expected_liveness):
        for output_idx, output_liveness in enumerate(outputs_liveness):
            w = self.path_weakrefs[depth][output_idx]
            if (stor_weak_ptr_and_data_ptr := maybe_deref(w)) is not None:
                assert output_liveness
                stor_weak_ptr, stor_data_ptr = stor_weak_ptr_and_data_ptr
                assert (stor_data_ptr in live_storage_data_ptrs) == (stor_weak_ptr in live_storage_weak_ptrs)
                live_storage_data_ptrs.add(stor_data_ptr)
                live_storage_weak_ptrs.add(stor_weak_ptr)
                is_persistent_alias = nodes[depth].static_output_tensors[output_idx] is not None
                if is_persistent_alias:
                    assert stor_data_ptr not in live_blocks
    for depth, output_index in newly_dead:
        assert not is_live(self.path_weakrefs[depth][output_index])