import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def referrednested(func, recurse=True):
    """get functions defined inside of func (e.g. inner functions in a closure)

    NOTE: results may differ if the function has been executed or not.
    If len(nestedcode(func)) > len(referrednested(func)), try calling func().
    If possible, python builds code objects, but delays building functions
    until func() is called.
    """
    import gc
    funcs = set()
    for co in nestedcode(func, recurse):
        for obj in gc.get_referrers(co):
            _ = getattr(obj, '__func__', None)
            if getattr(_, '__code__', None) is co:
                funcs.add(obj)
            elif getattr(obj, '__code__', None) is co:
                funcs.add(obj)
            elif getattr(obj, 'f_code', None) is co:
                funcs.add(obj)
            elif hasattr(obj, 'co_code') and obj is co:
                funcs.add(obj)
    return list(funcs)