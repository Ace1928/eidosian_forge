import types
import io as _io
def copy_module(mod, **defaults):
    copy = types.ModuleType(mod.__name__, doc=mod.__doc__)
    vars(copy).update(defaults)
    vars(copy).update(vars(mod))
    return copy