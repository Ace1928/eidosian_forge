from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.util.tf_export import tf_export
def smart_case(pred_fn_pairs, default=None, exclusive=False, name='smart_case'):
    """Like tf.case, except attempts to statically evaluate predicates.

  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the
  associated callable will be called or omitted depending on its value.
  Otherwise this functions like tf.case.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
    return control_flow_case._case_helper(smart_cond, pred_fn_pairs, default, exclusive, name, allow_python_preds=True)