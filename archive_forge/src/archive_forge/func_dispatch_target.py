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
@dispatch_for_api(api, {x_name: x_type, y_name: y_type})
def dispatch_target(*args, **kwargs):
    args, kwargs, name = _extract_name_arg(args, kwargs, name_index)
    if len(args) > 1:
        x, y, args = (args[0], args[1], args[2:])
    elif args:
        x, args = (args[0], args[1:])
        y = kwargs.pop(y_name, None)
    else:
        x = kwargs.pop(x_name, None)
        y = kwargs.pop(y_name, None)
    if need_to_bind_api_args:
        tensor_api = lambda v1, v2: api(v1, v2, *args, **kwargs)
    else:
        tensor_api = api
    if name is None:
        return elementwise_api_handler(tensor_api, x, y)
    else:
        with ops.name_scope(name, None, [x, y]):
            return elementwise_api_handler(tensor_api, x, y)