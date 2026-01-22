import re as _re
import sys as _sys
from tensorflow.python.util import tf_inspect as _tf_inspect
def reveal_undocumented(symbol_name, target_module=None):
    """Reveals a symbol that was previously removed by `remove_undocumented`.

  This should be used by tensorflow internal tests only. It explicitly
  defeats the encapsulation afforded by `remove_undocumented`.

  It throws an exception when the symbol was not hidden in the first place.

  Args:
    symbol_name: a string representing the full absolute path of the symbol.
    target_module: if specified, the module in which to restore the symbol.
  """
    if symbol_name not in _HIDDEN_ATTRIBUTES:
        raise LookupError('Symbol %s is not a hidden symbol' % symbol_name)
    symbol_basename = symbol_name.split('.')[-1]
    original_module, attr_value = _HIDDEN_ATTRIBUTES[symbol_name]
    if not target_module:
        target_module = original_module
    setattr(target_module, symbol_basename, attr_value)