import collections
import copy
import sys
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import layer_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.wrap', v1=[])
def wrap_or_unwrap(value):
    """Wraps input value into trackable data structures.

  This is mostly useful for containers like list, dict, etc, which could contain
  trackable objects in it. Wrapped data structure will be tracked when
  associated with a `tf.Module`, so that save model/checkpoint can properly
  track the dependency.

  It will also unwrap NoDependency objects.

  Args:
    value: the input object to be wrapped.

  Returns:
    Wrapped trackable data structure.
  """
    if isinstance(value, NoDependency):
        return value.value
    if isinstance(value, base.Trackable):
        return value
    elif type(value) == dict:
        return _DictWrapper(value)
    elif type(value) == collections.OrderedDict:
        return _DictWrapper(value)
    elif type(value) == list:
        return ListWrapper(value)
    elif isinstance(value, tuple) and _should_wrap_tuple(value):
        return _TupleWrapper(value)
    else:
        return value