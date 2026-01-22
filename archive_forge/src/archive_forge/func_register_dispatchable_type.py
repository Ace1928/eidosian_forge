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
def register_dispatchable_type(cls):
    """Class decorator that registers a type for use with type-based dispatch.

  Should *not* be used with subclasses of `CompositeTensor` or `ExtensionType`
  (which are automatically registered).

  Note: this function is intended to support internal legacy use cases (such
  as RaggedTensorValue), and will probably not be exposed as a public API.

  Args:
    cls: The class to register.

  Returns:
    `cls`.
  """
    _api_dispatcher.register_dispatchable_type(cls)
    return cls