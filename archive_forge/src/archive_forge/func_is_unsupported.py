import functools
import inspect
import sys
import unittest
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.util import tf_inspect
def is_unsupported(o):
    """Checks whether an entity is supported by AutoGraph at all."""
    if _is_known_loaded_type(o, 'wrapt', 'FunctionWrapper') or _is_known_loaded_type(o, 'wrapt', 'BoundFunctionWrapper'):
        logging.warning('{} appears to be decorated by wrapt, which is not yet supported by AutoGraph. The function will run as-is. You may still apply AutoGraph before the wrapt decorator.'.format(o))
        logging.log(2, 'Permanently allowed: %s: wrapt decorated', o)
        return True
    if _is_known_loaded_type(o, 'functools', '_lru_cache_wrapper'):
        logging.log(2, 'Permanently allowed: %s: lru_cache', o)
        return True
    if inspect_utils.isconstructor(o):
        logging.log(2, 'Permanently allowed: %s: constructor', o)
        return True
    if any((_is_of_known_loaded_module(o, m) for m in ('collections', 'pdb', 'copy', 'inspect', 're'))):
        logging.log(2, 'Permanently allowed: %s: part of builtin module', o)
        return True
    if hasattr(o, '__module__') and hasattr(o.__module__, '_IS_TENSORFLOW_PLUGIN'):
        logging.log(2, 'Permanently allowed: %s: TensorFlow plugin', o)
        return True
    return False