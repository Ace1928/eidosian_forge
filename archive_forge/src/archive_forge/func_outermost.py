import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def outermost(func):
    """get outermost enclosing object (i.e. the outer function in a closure)

    NOTE: this is the object-equivalent of getsource(func, enclosing=True)
    """
    if ismethod(func):
        _globals = func.__func__.__globals__ or {}
    elif isfunction(func):
        _globals = func.__globals__ or {}
    else:
        return
    _globals = _globals.items()
    from .source import getsourcelines
    try:
        lines, lnum = getsourcelines(func, enclosing=True)
    except Exception:
        lines, lnum = ([], None)
    code = ''.join(lines)
    _locals = ((name, obj) for name, obj in _globals if name in code)
    for name, obj in _locals:
        try:
            if getsourcelines(obj) == (lines, lnum):
                return obj
        except Exception:
            pass
    return