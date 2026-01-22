import inspect
import os
import re
import textwrap
import typing
from typing import Union
import warnings
from collections import OrderedDict
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp
from rpy2.robjects import help
from rpy2.robjects import conversion
from rpy2.robjects.vectors import Vector
from rpy2.robjects.packages_utils import (default_symbol_r2python,
def wrap_r_function(r_func: SignatureTranslatedFunction, name: str, *, is_method: bool=False, full_repr: bool=False, map_default: typing.Optional[typing.Callable[[rinterface.Sexp], typing.Any]]=_map_default_value, wrap_docstring: typing.Optional[typing.Callable[[SignatureTranslatedFunction, bool, inspect.Signature, typing.Optional[int]], str]]=wrap_docstring_default) -> typing.Callable:
    """
    Wrap an rpy2 function handle with a Python function of matching signature.

    Args:
        r_func (rpy2.robjects.functions.SignatureTranslatedFunction): The
        function to be wrapped.
        name (str): The name of the function.
        is_method (bool): Whether the function should be treated as a method
        (adds a `self` param to the signature if so).
        map_default (function): Function to map default values in the Python
        signature. No mapping to default values is done if None.
    Returns:
        A function wrapping an underlying R function.
    """
    name = name.replace('.', '_')
    signature, r_ellipsis = map_signature(r_func, is_method=is_method, map_default=map_default)
    if r_ellipsis:

        def wrapped_func(*args, **kwargs):
            new_args = list(((None, x) for x in rinterface.args[:r_ellipsis])) + list(args[r_ellipsis]) + list(((None, x) for x in args[min(r_ellipsis + 1, len(args) - 1):])) + list(kwargs.items())
            value = r_func.rcall(new_args)
            return value
    else:

        def wrapped_func(*args, **kwargs):
            value = r_func(*args, **kwargs)
            return value
    if wrap_docstring:
        docstring = wrap_docstring(r_func, is_method, signature, r_ellipsis)
    else:
        docstring = 'This is a dynamically created wrapper for an R function.'
    wrapped_func.__name__ = name
    wrapped_func.__qualname__ = name
    wrapped_func.__signature__ = signature
    wrapped_func.__doc__ = docstring
    return wrapped_func