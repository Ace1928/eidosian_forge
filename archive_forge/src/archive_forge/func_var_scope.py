import functools
import traceback
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@property
@deprecated('2017-02-21', 'The .var_scope property is deprecated. Please change your code to use the .variable_scope property')
def var_scope(self):
    """Returns the variable scope object created by this Template."""
    return self._variable_scope