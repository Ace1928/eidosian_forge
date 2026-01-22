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
def instantiate(self, globals_, closure, defaults=None, kwdefaults=None):
    """Creates a new function instance."""
    if self._unbound_factory is None:
        raise ValueError('call create first')
    factory_code = self._unbound_factory.__code__
    factory_freevars = factory_code.co_freevars
    closure_map = dict(zip(self._freevars, closure))
    factory_closure = tuple((closure_map[name] for name in factory_code.co_freevars))
    if len(factory_closure) != len(closure):
        raise ValueError('closure mismatch, requested {}, but source function had {}'.format(self._freevars, factory_freevars))
    bound_factory = types.FunctionType(code=factory_code, globals=globals_, name=self._name, argdefs=(), closure=factory_closure)
    new_fn = bound_factory(**self._extra_locals)
    if defaults:
        new_fn.__defaults__ = defaults
    if kwdefaults:
        new_fn.__kwdefaults__ = kwdefaults
    return new_fn