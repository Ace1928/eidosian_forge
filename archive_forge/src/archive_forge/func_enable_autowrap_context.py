import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
@staticmethod
def enable_autowrap_context(kwargs: Any) -> None:
    if _ConfigAutoWrap.in_autowrap_context:
        raise NotImplementedError('You are already within an autowrap context and we currently do not supported nested autowrap.')
    _ConfigAutoWrap.in_autowrap_context = True
    assert 'wrapper_cls' in kwargs.keys(), 'Expected to pass in wrapper_cls arg into _ConfigAutoWrap.'
    _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs['wrapper_cls'])
    del kwargs['wrapper_cls']
    _ConfigAutoWrap.kwargs = kwargs