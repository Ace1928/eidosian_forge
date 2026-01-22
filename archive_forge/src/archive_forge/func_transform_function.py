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
def transform_function(self, fn, user_context):
    """Transforms a function. See GenericTranspiler.trasnform_function.

    This overload wraps the parent's `transform_function`, adding caching and
    facilities to instantiate the output as a Python object. It also
    adds facilities to make new symbols available to the generated Python code,
    visible as local variables - see `get_extra_locals`.

    Args:
      fn: A function or lambda.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user attribute.
    Returns:
      A tuple:
        * A function or lambda with the same signature and closure as `fn`
        * The temporary module into which the transformed function was loaded
        * The source map as a
            Dict[origin_info.LineLocation, origin_info.OriginInfo]
    """
    cache_subkey = self.get_caching_key(user_context)
    if self._cache.has(fn, cache_subkey):
        factory = self._cached_factory(fn, cache_subkey)
    else:
        with self._cache_lock:
            if self._cache.has(fn, cache_subkey):
                factory = self._cached_factory(fn, cache_subkey)
            else:
                logging.log(1, '%s is not cached for subkey %s', fn, cache_subkey)
                nodes, ctx = super(PyToPy, self).transform_function(fn, user_context)
                if isinstance(nodes, gast.Lambda):
                    nodes = gast.Assign(targets=[gast.Name(ctx.info.name, ctx=gast.Store(), annotation=None, type_comment=None)], value=nodes)
                else:
                    nodes.name = ctx.info.name
                if logging.has_verbosity(2):
                    logging.log(2, 'Transformed %s:\n\n%s\n', fn, parser.unparse(nodes))
                factory = _PythonFnFactory(ctx.info.name, fn.__code__.co_freevars, self.get_extra_locals())
                factory.create(nodes, ctx.namer, future_features=ctx.info.future_features)
                self._cache[fn][cache_subkey] = factory
    transformed_fn = factory.instantiate(globals_=fn.__globals__, closure=fn.__closure__ or (), defaults=fn.__defaults__, kwdefaults=getattr(fn, '__kwdefaults__', None))
    return (transformed_fn, factory.module, factory.source_map)