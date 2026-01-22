import inspect
from typing import Callable, Optional

    Args:
        hook_fx: the hook callable
        param: the name of the parameter to check
        explicit: whether the parameter has to be explicitly declared
        min_args: whether the `signature` has at least `min_args` parameters
    