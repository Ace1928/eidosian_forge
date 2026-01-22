import json
import inspect
from lazyops.utils.helpers import is_coro_func
from typing import Callable, Optional, List
def create_post_function_hook_wrapper(function: Callable):
    """
    Creates a function wrapper that executes after the function
    """

    def inner_wrapper(handler: Callable):
        """
        The inner wrapper
        """

        async def wrapper(*args, **kwargs):
            result = await handler(*args, **kwargs)
            if is_coro_func(function):
                await function(result, *args, **kwargs)
            else:
                function(result, *args, **kwargs)
            return result
        return wrapper
    return inner_wrapper