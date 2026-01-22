import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def check_concat_weights(match):
    weights = [match.kwargs['w1'], match.kwargs['w2']]
    if 'w3' in match.kwargs:
        weights.append(match.kwargs['w3'])
    return all((w.op == 'get_attr' and w.meta['val'].shape == weights[0].meta['val'].shape for w in weights))