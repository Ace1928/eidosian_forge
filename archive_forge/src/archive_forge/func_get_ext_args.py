import copy
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
def get_ext_args(**kwargs: Any):
    """Validate and convert arguments. Reused in Doc, Token and Span."""
    default = kwargs.get('default')
    getter = kwargs.get('getter')
    setter = kwargs.get('setter')
    method = kwargs.get('method')
    if getter is None and setter is not None:
        raise ValueError(Errors.E089)
    valid_opts = ('default' in kwargs, method is not None, getter is not None)
    nr_defined = sum((t is True for t in valid_opts))
    if nr_defined != 1:
        raise ValueError(Errors.E083.format(nr_defined=nr_defined))
    if setter is not None and (not hasattr(setter, '__call__')):
        raise ValueError(Errors.E091.format(name='setter', value=repr(setter)))
    if getter is not None and (not hasattr(getter, '__call__')):
        raise ValueError(Errors.E091.format(name='getter', value=repr(getter)))
    if method is not None and (not hasattr(method, '__call__')):
        raise ValueError(Errors.E091.format(name='method', value=repr(method)))
    return (default, method, getter, setter)