from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables
def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
    if isinstance(pred, variables.Variable):
        return cond.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)