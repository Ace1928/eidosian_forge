import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def client_mode_hook(func: callable):
    """Decorator for whether to use the 'regular' ray version of a function,
    or the Ray Client version of that function.

    Args:
        func: This function. This is set when this function is used
            as a decorator.
    """
    from ray.util.client import ray

    @wraps(func)
    def wrapper(*args, **kwargs):
        if client_mode_should_convert():
            if func.__name__ != 'init' or is_client_mode_enabled_by_default:
                return getattr(ray, func.__name__)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper