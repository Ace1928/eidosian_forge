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
def replace_iterable_params(args, kwargs, iterable_params):
    """Returns (args, kwargs) with any iterable parameters converted to lists.

  Args:
    args: Positional rguments to a function
    kwargs: Keyword arguments to a function.
    iterable_params: A list of (name, index) tuples for iterable parameters.

  Returns:
    A tuple (args, kwargs), where any positional or keyword parameters in
    `iterable_params` have their value converted to a `list`.
  """
    args = list(args)
    for name, index in iterable_params:
        if index < len(args):
            args[index] = list(args[index])
        elif name in kwargs:
            kwargs[name] = list(kwargs[name])
    return (tuple(args), kwargs)