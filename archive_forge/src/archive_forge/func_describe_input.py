import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f'parameter/buffer {i}'
    else:
        return f'input {i - aot_config.num_params_buffers}'