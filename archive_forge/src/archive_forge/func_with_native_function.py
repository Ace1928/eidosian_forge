import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import torchgen.local as local
from torchgen.model import (
from torchgen.utils import context, S, T
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:

    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)
    return wrapper