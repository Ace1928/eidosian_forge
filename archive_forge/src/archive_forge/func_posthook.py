import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def posthook(grad_input, grad_output):
    fx_traceback.set_stack_trace(special_stack_)
    fx_traceback.reset_grad_fn_seq_nr()