from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
def resolve_target_str(target_str):
    """Resolves a target specified as a string to its Target class."""
    return target_registry[target_str]