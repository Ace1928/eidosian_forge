import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def compatible_func(*args, **kwargs):
    bound = op_signature.bind(*args, **kwargs)
    for name, param in func_missing_params.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments.pop(name)
        if value is not param.default:
            raise AssertionError(f'Dispatched op is called with argument `{name}` set to a non-default value, which is not supported by the decorated function')
    return func(*bound.args, **bound.kwargs)