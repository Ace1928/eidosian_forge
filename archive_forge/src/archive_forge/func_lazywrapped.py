from __future__ import annotations
import functools
from lazyops.libs.pooler import is_coro_func, ThreadPooler
from typing import TypeVar, Callable, Any
def lazywrapped(func: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    """
        Wrapper Function
        """
    if is_coro_func(func):

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs) -> ReturnT:
            """
                Wrapped Function
                """
            nonlocal _initialized_function, _initialized
            if not _initialized:
                _initialized_function = await ThreadPooler.asyncish(function, *function_args, **function_kwargs)
                _initialized = True
            if _initialized_function is None:
                return await func(*args, **kwargs)
            return await _initialized_function(func)(*args, **kwargs)
        return _wrapper

    @functools.wraps(func)
    def _wrapper(*args, **kwargs) -> ReturnT:
        """
            Wrapped Function
            """
        nonlocal _initialized_function, _initialized
        if not _initialized:
            _initialized_function = function(*function_args, **function_kwargs)
            _initialized = True
        if _initialized_function is None:
            return func(*args, **kwargs)
        return _initialized_function(func)(*args, **kwargs)
    return _wrapper