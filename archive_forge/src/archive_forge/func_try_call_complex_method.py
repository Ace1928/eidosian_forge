from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def try_call_complex_method(cls_type, method):
    """ __complex__ needs special treatment as the argument names are kwargs
    and therefore specific in name and default value.
    """
    if method in cls_type.jit_methods:
        template = f'\ndef func(real=0, imag=0):\n    return real.{method}()\n'
        return extract_template(template, 'func')