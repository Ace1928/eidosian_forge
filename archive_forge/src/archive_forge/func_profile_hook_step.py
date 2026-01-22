import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
@staticmethod
def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]:

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> R:
        self, *_ = args
        self = cast(Optimizer, self)
        profile_name = f'Optimizer.step#{self.__class__.__name__}.step'
        with torch.autograd.profiler.record_function(profile_name):
            for pre_hook in chain(_global_optimizer_pre_hooks.values(), self._optimizer_step_pre_hooks.values()):
                result = pre_hook(self, args, kwargs)
                if result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        args, kwargs = result
                    else:
                        raise RuntimeError(f'{func} must return None or a tuple of (new_args, new_kwargs), but got {result}.')
            out = func(*args, **kwargs)
            self._optimizer_step_code()
            for post_hook in chain(self._optimizer_step_post_hooks.values(), _global_optimizer_post_hooks.values()):
                post_hook(self, args, kwargs)
            return out
    return wrapper