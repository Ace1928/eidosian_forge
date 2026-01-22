import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode
from .pointers import parent, reference, at, parents, children
from .logger import trace
def baditems(obj, exact=False, safe=False):
    """get items in object that fail to pickle"""
    if not hasattr(obj, '__iter__'):
        return [j for j in (badobjects(obj, 0, exact, safe),) if j is not None]
    obj = obj.values() if getattr(obj, 'values', None) else obj
    _obj = []
    [_obj.append(badobjects(i, 0, exact, safe)) for i in obj if i not in _obj]
    return [j for j in _obj if j is not None]