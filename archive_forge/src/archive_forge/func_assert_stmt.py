from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.util import tf_inspect
def assert_stmt(expression1, expression2):
    """Functional form of an assert statement.

  This follows the semantics of the Python assert statement, however the
  concrete implementations may deviate from it. See the respective
  implementation for details.

  In general, the assert statement should not be used for control flow.
  Furthermore, it is encouraged that the assertion expressions should not have
  side effects.

  Args:
    expression1: Any
    expression2: Callable[[], Any], returns the expression to include in the
        error message when expression1 evaluates to False. When expression1 is
        True, the result of expression2 will not be evaluated, however,
        expression2 itself may be evaluated in some implementations.

  Returns:
    Any, implementation-dependent.

  Raises:
    ValueError: if any arguments are illegal.
  """
    if not callable(expression2):
        raise ValueError('{} must be a callable'.format(expression2))
    args, _, keywords, _ = tf_inspect.getargspec(expression2)
    if args or keywords:
        raise ValueError('{} may not have any arguments'.format(expression2))
    if tensor_util.is_tf_type(expression1):
        return _tf_assert_stmt(expression1, expression2)
    else:
        return _py_assert_stmt(expression1, expression2)