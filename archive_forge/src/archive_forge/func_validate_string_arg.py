import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def validate_string_arg(input_data, allowable_strings, layer_name, arg_name, allow_none=False, allow_callables=False):
    """Validates the correctness of a string-based arg."""
    if allow_none and input_data is None:
        return
    elif allow_callables and callable(input_data):
        return
    elif isinstance(input_data, str) and input_data in allowable_strings:
        return
    else:
        allowed_args = '`None`, ' if allow_none else ''
        allowed_args += 'a `Callable`, ' if allow_callables else ''
        allowed_args += 'or one of the following values: %s' % (allowable_strings,)
        raise ValueError('The %s argument of layer %s received an invalid value %s. Allowed values are: %s.' % (arg_name, layer_name, input_data, allowed_args))