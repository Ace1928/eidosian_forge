import re as _re
import sys as _sys
from tensorflow.python.util import tf_inspect as _tf_inspect
def make_all(module_name, doc_string_modules=None):
    """Generates `__all__` from the docstring of one or more modules.

  Usage: `make_all(__name__)` or
  `make_all(__name__, [sys.modules(__name__), other_module])`. The doc string
  modules must each a docstring, and `__all__` will contain all symbols with
  `@@` references, where that symbol currently exists in the module named
  `module_name`.

  Args:
    module_name: The name of the module (usually `__name__`).
    doc_string_modules: a list of modules from which to take docstring.
    If None, then a list containing only the module named `module_name` is used.

  Returns:
    A list suitable for use as `__all__`.
  """
    if doc_string_modules is None:
        doc_string_modules = [_sys.modules[module_name]]
    cur_members = set((name for name, _ in _tf_inspect.getmembers(_sys.modules[module_name])))
    results = set()
    for doc_module in doc_string_modules:
        results.update([m.group(1) for m in _reference_pattern.finditer(doc_module.__doc__) if m.group(1) in cur_members])
    return list(results)