import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def badtypes(obj, depth=0, exact=False, safe=False):
    """get types for objects that fail to pickle"""
    from dill import pickles
    if not depth:
        if pickles(obj, exact, safe):
            return None
        return type(obj)
    return dict(((attr, badtypes(getattr(obj, attr), depth - 1, exact, safe)) for attr in dir(obj) if not pickles(getattr(obj, attr), exact, safe)))