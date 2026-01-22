import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import torchgen.local as local
from torchgen.model import (
from torchgen.utils import context, S, T
def method_with_nested_native_function(func: Callable[[S, F3], T]) -> Callable[[S, F3], T]:

    @functools.wraps(func)
    def wrapper(slf: S, f: F3) -> T:
        with native_function_manager(f[0]):
            return func(slf, f)
    return wrapper