import functools
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.common_op_utils import _basic_validation
def decorator_sharded_func(wrapped_func):

    @functools.wraps(wrapped_func)
    def wrapper(types, args=(), kwargs=None, pg=None):
        _basic_validation(op, args, kwargs)
        st = args[0]
        if kwargs is None:
            kwargs = {}
        if extra_check:
            extra_check(*args, **kwargs)
        if early_stop_func:
            early_stop = early_stop_func(*args, **kwargs)
            if early_stop:
                return st
        return wrapped_func(types, args, kwargs, pg)
    return wrapper