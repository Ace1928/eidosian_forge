import inspect
import functools
from enum import Enum
import torch.autograd
@functools.wraps(next_func)
def wrap_next(*args, **kwargs):
    datapipe = args[0]
    if torch.autograd._profiler_enabled():
        with profiler_record_fn_context(datapipe):
            result = next_func(*args, **kwargs)
    else:
        result = next_func(*args, **kwargs)
    datapipe._number_of_samples_yielded += 1
    return result