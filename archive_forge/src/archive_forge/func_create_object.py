import functools
import inspect
import itertools
import logging
import sys
import threading
import types
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import (
def create_object(self, cls: Type[T], additional_kwargs: Any=None) -> T:
    """Create a new instance, satisfying any dependencies on cls."""
    additional_kwargs = additional_kwargs or {}
    log.debug('%sCreating %r object with %r', self._log_prefix, cls, additional_kwargs)
    try:
        instance = cls.__new__(cls)
    except TypeError as e:
        reraise(e, CallError(cls, getattr(cls.__new__, '__func__', cls.__new__), (), {}, e, self._stack), maximum_frames=2)
    init = cls.__init__
    try:
        self.call_with_injection(init, self_=instance, kwargs=additional_kwargs)
    except TypeError as e:
        init_function = instance.__init__.__func__
        reraise(e, CallError(instance, init_function, (), additional_kwargs, e, self._stack))
    return instance