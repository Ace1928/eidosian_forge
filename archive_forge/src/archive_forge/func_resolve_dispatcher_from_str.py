from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
def resolve_dispatcher_from_str(target_str):
    """Returns the dispatcher associated with a target string"""
    target_hw = resolve_target_str(target_str)
    return dispatcher_registry[target_hw]