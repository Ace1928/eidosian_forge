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
def run_eager(self, new_inputs, function_id: FunctionID):
    already_warm = function_id in self.warmed_up_functions
    if not already_warm:
        log.debug('Running warmup of function %d', function_id.id)
    else:
        log.debug('Running eager of function %d because ancestor needed to warm up', function_id.id)
    self.warmed_up_functions.add(function_id)
    node = CUDAWarmupNode(self.ids_to_funcs[function_id], self.current_node, self.cuda_graphs_thread_pool, self.graph, self.device_index, self.ids_to_stack_traces[function_id], self.stream, already_warm)
    self.current_node = node
    self.path_state = ExecutionState.WARMUP
    self.update_generation()
    return node.run(new_inputs)