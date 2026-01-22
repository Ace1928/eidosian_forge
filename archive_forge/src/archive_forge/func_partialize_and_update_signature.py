import functools
import inspect
import logging
import math
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
def partialize_and_update_signature(func, **kwargs):
    """
    Equivalent to functools.partial but also updates the signature on returned function
    """
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters
    new_parameters = {key: value for key, value in parameters.items() if key not in kwargs}
    new_sig = inspect.Signature(parameters=list(new_parameters.values()))
    partial_func = functools.partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        return partial_func(*args, **kwargs)
    wrapper.__signature__ = new_sig
    wrapper.__name__ = func.__name__
    return wrapper