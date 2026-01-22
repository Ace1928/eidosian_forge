from functools import wraps
from jedi import debug
def inference_state_function_cache(default=_NO_DEFAULT):

    def decorator(func):
        return _memoize_default(default=default, inference_state_is_first_arg=True)(func)
    return decorator