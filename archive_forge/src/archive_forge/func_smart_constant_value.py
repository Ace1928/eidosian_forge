from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.util.tf_export import tf_export
def smart_constant_value(pred):
    """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Args:
    pred: A scalar, either a Python bool or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Tensor or bool.
  """
    if isinstance(pred, tensor.Tensor):
        pred_value = tensor_util.constant_value(pred)
        if pred_value is None:
            pred_value = tensor_util.try_evaluate_constant(pred)
    elif pred in {0, 1}:
        pred_value = bool(pred)
    elif isinstance(pred, bool):
        pred_value = pred
    else:
        raise TypeError(f'Argument `pred` must be a Tensor, or a Python bool, or 1 or 0. Received: pred={pred} of type {type(pred).__name__}')
    return pred_value