import inspect
from typing import Callable, Optional
def is_param_in_hook_signature(hook_fx: Callable, param: str, explicit: bool=False, min_args: Optional[int]=None) -> bool:
    """
    Args:
        hook_fx: the hook callable
        param: the name of the parameter to check
        explicit: whether the parameter has to be explicitly declared
        min_args: whether the `signature` has at least `min_args` parameters
    """
    if hasattr(hook_fx, '__wrapped__'):
        hook_fx = hook_fx.__wrapped__
    parameters = inspect.getfullargspec(hook_fx)
    args = parameters.args[1:]
    return param in args or (not explicit and parameters.varargs is not None) or (isinstance(min_args, int) and len(args) >= min_args)