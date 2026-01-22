import importlib
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2
def has_deprecation_decorator(symbol):
    """Checks if given object has a deprecation decorator.

  We check if deprecation decorator is in decorators as well as
  whether symbol is a class whose __init__ method has a deprecation
  decorator.
  Args:
    symbol: Python object.

  Returns:
    True if symbol has deprecation decorator.
  """
    decorators, symbol = tf_decorator.unwrap(symbol)
    if contains_deprecation_decorator(decorators):
        return True
    if tf_inspect.isfunction(symbol):
        return False
    if not tf_inspect.isclass(symbol):
        return False
    if not hasattr(symbol, '__init__'):
        return False
    init_decorators, _ = tf_decorator.unwrap(symbol.__init__)
    return contains_deprecation_decorator(init_decorators)