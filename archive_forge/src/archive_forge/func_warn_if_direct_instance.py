import contextlib
import functools
import inspect
import warnings
from typing import Any, Callable, Generator, Type, TypeVar, Union, cast
from langchain_core._api.internal import is_caller_internal
def warn_if_direct_instance(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Warn that the class is in beta."""
    nonlocal warned
    if not warned and type(self) is obj and (not is_caller_internal()):
        warned = True
        emit_warning()
    return wrapped(self, *args, **kwargs)