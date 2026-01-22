import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def register_binary_folding_pattern(pattern, extra_check=_return_true):
    return register_graph_pattern(pattern, extra_check=extra_check, pass_dict=binary_folding_pass)