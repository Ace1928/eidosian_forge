import inspect
import threading
import types
import gast
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
def transform_module(self, mod, user_context):
    """Transforms a module.

    Subclasses may override this method. The return value is opaque.

    The method receives the original AST. The result is passed as-is to the
    output of `transform`.

    Args:
      mod: A Python module.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      List[Tuple[Any, Any]]. By default it returns the output of transform_ast,
      evaluated on each supported member, other than modules, together with a
      `transformer.Context` containing information about the transformation
      process.
    """
    result = []
    for member in mod.__dict__.values():
        if inspect.ismodule(member):
            continue
        try:
            result.append(self.transform(member, user_context))
        except NotImplementedError:
            pass
    return result