import torch
from .. import Config, autotune, cdiv, heuristics, jit
from .. import language as tl
from .matmul_perf_model import early_config_prune, estimate_matmul_time
def get_higher_dtype(a, b):
    if a is b:
        return a
    assert a in _ordered_datatypes
    assert b in _ordered_datatypes
    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a