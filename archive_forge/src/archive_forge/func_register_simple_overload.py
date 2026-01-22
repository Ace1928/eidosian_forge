from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def register_simple_overload(func, *attrs, n_args=1):
    """
    Register an overload for func that checks for methods __attr__ for each
    attr in attrs.
    """
    arg_names = _get_args(n_args)
    template = f'\ndef func({','.join(arg_names)}):\n    pass\n'

    @wraps(extract_template(template, 'func'))
    def overload_func(*args, **kwargs):
        options = [try_call_method(args[0], f'__{attr}__', n_args) for attr in attrs]
        return take_first(*options)
    return class_instance_overload(func)(overload_func)