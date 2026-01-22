import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def trace_model_call(model, input_signature=None):
    """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
    if input_signature is None:
        if isinstance(model.call, def_function.Function):
            input_signature = model.call.input_signature
    if input_signature is None:
        input_signature = model_input_signature(model)
    if input_signature is None:
        raise_model_input_error(model)

    @def_function.function(input_signature=input_signature)
    def _wrapped_model(*args):
        """A concrete tf.function that wraps the model's call function."""
        inputs = args[0] if len(input_signature) == 1 else list(args)
        with base_layer_utils.call_context().enter(model, inputs=inputs, build_graph=False, training=False, saving=True):
            outputs = model(inputs, training=False)
        output_names = model.output_names
        if output_names is None:
            from tensorflow.python.keras.engine import compile_utils
            output_names = compile_utils.create_pseudo_output_names(outputs)
        outputs = nest.flatten(outputs)
        return {name: output for name, output in zip(output_names, outputs)}
    return _wrapped_model