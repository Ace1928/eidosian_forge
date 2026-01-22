import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def badobjects(obj, depth=0, exact=False, safe=False):
    """get objects that fail to pickle"""
    from dill import pickles
    if not depth:
        if pickles(obj, exact, safe):
            return None
        return obj
    return dict(((attr, badobjects(getattr(obj, attr), depth - 1, exact, safe)) for attr in dir(obj) if not pickles(getattr(obj, attr), exact, safe)))