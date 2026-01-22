from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
@contextmanager
def setup_rnn(use_input_variant, args, kwargs):
    with batch_second(args, kwargs) if use_input_variant else allow_smaller_batches(args, kwargs):
        yield