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
def maybe_add_training_arg(original_call, wrapped_call, expects_training_arg, default_training_value):
    """Decorate call and optionally adds training argument.

  If a layer expects a training argument, this function ensures that 'training'
  is present in the layer args or kwonly args, with the default training value.

  Args:
    original_call: Original call function.
    wrapped_call: Wrapped call function.
    expects_training_arg: Whether to include 'training' argument.
    default_training_value: Default value of the training kwarg to include in
      the arg spec. If `None`, the default is `K.learning_phase()`.

  Returns:
    Tuple of (
      function that calls `wrapped_call` and sets the training arg,
      Argspec of returned function or `None` if the argspec is unchanged)
  """
    if not expects_training_arg:
        return (wrapped_call, None)

    def wrap_with_training_arg(*args, **kwargs):
        """Wrap the `wrapped_call` function, and set training argument."""
        training_arg_index = get_training_arg_index(original_call)
        training = get_training_arg(training_arg_index, args, kwargs)
        if training is None:
            training = default_training_value or K.learning_phase()
        args = list(args)
        kwargs = kwargs.copy()

        def replace_training_and_call(training):
            set_training_arg(training, training_arg_index, args, kwargs)
            return wrapped_call(*args, **kwargs)
        return control_flow_util.smart_cond(training, lambda: replace_training_and_call(True), lambda: replace_training_and_call(False))
    arg_spec = tf_inspect.getfullargspec(original_call)
    defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []
    kwonlyargs = arg_spec.kwonlyargs
    kwonlydefaults = arg_spec.kwonlydefaults or {}
    if 'training' not in arg_spec.args:
        kwonlyargs.append('training')
        kwonlydefaults['training'] = default_training_value
    else:
        index = arg_spec.args.index('training')
        training_default_index = len(arg_spec.args) - index
        if arg_spec.defaults and len(arg_spec.defaults) >= training_default_index and (defaults[-training_default_index] is None):
            defaults[-training_default_index] = default_training_value
    decorator_argspec = tf_inspect.FullArgSpec(args=arg_spec.args, varargs=arg_spec.varargs, varkw=arg_spec.varkw, defaults=defaults, kwonlyargs=kwonlyargs, kwonlydefaults=kwonlydefaults, annotations=arg_spec.annotations)
    return (wrap_with_training_arg, decorator_argspec)