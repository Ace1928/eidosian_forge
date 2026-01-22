import inspect
import sys
from .__wrapt__ import FunctionWrapper
def wrap_object(module, name, factory, args=(), kwargs={}):
    parent, attribute, original = resolve_path(module, name)
    wrapper = factory(original, *args, **kwargs)
    apply_patch(parent, attribute, wrapper)
    return wrapper