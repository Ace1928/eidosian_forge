import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def res_name(self, ns, types_ns, name):
    """Resolves the type/value an external (e.g. closure, global) variable.

    Args:
      ns: namespace
      types_ns: types namespace
      name: symbol name
    Returns:
      Tuple (type, static_value). The first element is the type to use for
      inferrence. The second is the static value to use. Return None to treat it
      as unknown.
    """
    raise NotImplementedError('subclasses must implement')