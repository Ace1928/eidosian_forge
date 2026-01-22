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
def contains_cls(x):
    """Returns true if `x` contains `cls`."""
    if isinstance(x, dict):
        return any((contains_cls(v) for v in x.values()))
    elif x is cls:
        return True
    elif type_annotations.is_generic_list(x) or type_annotations.is_generic_union(x):
        type_args = type_annotations.get_generic_type_args(x)
        return any((contains_cls(arg) for arg in type_args))
    else:
        return False