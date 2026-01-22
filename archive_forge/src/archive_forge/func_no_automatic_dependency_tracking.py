import collections
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.trackable import constants
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.tracking.no_automatic_dependency_tracking', v1=[])
def no_automatic_dependency_tracking(method):
    """Disables automatic dependency tracking on attribute assignment.

  Use to decorate any method of a Trackable object. Attribute assignment in
  that method will not add dependencies (also respected in Model). Harmless if
  used in a class which does not do automatic dependency tracking (which means
  it's safe to use in base classes which may have subclasses which also inherit
  from Trackable).

  Args:
    method: The method to decorate.

  Returns:
    A decorated method which sets and un-sets automatic dependency tracking for
    the object the method is called on (not thread safe).
  """

    def _method_wrapper(self, *args, **kwargs):
        previous_value = getattr(self, '_self_setattr_tracking', True)
        self._self_setattr_tracking = False
        try:
            result = method(self, *args, **kwargs)
        finally:
            self._self_setattr_tracking = previous_value
        return result
    return tf_decorator.make_decorator(target=method, decorator_func=_method_wrapper)