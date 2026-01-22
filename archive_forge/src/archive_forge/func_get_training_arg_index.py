import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
def get_training_arg_index(call_fn):
    """Returns the index of 'training' in the layer call function arguments.

  Args:
    call_fn: Call function.

  Returns:
    - n: index of 'training' in the call function arguments.
    - -1: if 'training' is not found in the arguments, but layer.call accepts
          variable keyword arguments
    - None: if layer doesn't expect a training argument.
  """
    argspec = tf_inspect.getfullargspec(call_fn)
    if argspec.varargs:
        if 'training' in argspec.kwonlyargs or argspec.varkw:
            return -1
        return None
    else:
        arg_list = argspec.args
        if tf_inspect.ismethod(call_fn):
            arg_list = arg_list[1:]
        if 'training' in arg_list:
            return arg_list.index('training')
        elif 'training' in argspec.kwonlyargs or argspec.varkw:
            return -1
        return None