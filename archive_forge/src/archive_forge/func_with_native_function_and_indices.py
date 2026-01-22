import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import torchgen.local as local
from torchgen.model import (
from torchgen.utils import context, S, T
def with_native_function_and_indices(func: Callable[[F, Dict[DispatchKey, BackendIndex]], T]) -> Callable[[F, Dict[DispatchKey, BackendIndex]], T]:

    @functools.wraps(func)
    def wrapper(f: F, backend_indices: Dict[DispatchKey, BackendIndex]) -> T:
        with native_function_manager(f):
            return func(f, backend_indices)
    return wrapper