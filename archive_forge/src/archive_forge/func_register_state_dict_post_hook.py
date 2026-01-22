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
def register_state_dict_post_hook(self, hook: Callable[['Optimizer', StateDict], Optional[StateDict]], prepend: bool=False) -> RemovableHandle:
    """Register a state dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
    handle = hooks.RemovableHandle(self._optimizer_state_dict_post_hooks)
    self._optimizer_state_dict_post_hooks[handle.id] = hook
    if prepend:
        self._optimizer_state_dict_post_hooks.move_to_end(handle.id, last=False)
    return handle