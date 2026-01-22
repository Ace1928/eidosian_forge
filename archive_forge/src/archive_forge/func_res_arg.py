import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def res_arg(self, ns, types_ns, f_name, name, type_anno, f_is_local):
    """Resolves the type of a (possibly annotated) function argument.

    Args:
      ns: namespace
      types_ns: types namespace
      f_name: str, the function name
      name: str, the argument name
      type_anno: the type annotating the argument, if any
      f_is_local: bool, whether the function is a local function
    Returns:
      Set of the argument types.
    """
    raise NotImplementedError('subclasses must implement')