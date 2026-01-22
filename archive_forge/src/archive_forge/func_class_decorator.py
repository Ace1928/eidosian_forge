import functools
import inspect
import sys
from pyomo.common import DeveloperError
def class_decorator(cls):
    assert len(cls.__bases__) == 1
    base = cls.__bases__[0]

    def construct(self, data=None):
        if hasattr(self, '_name') and self._name == self.__class__.__name__:
            self._name = base.__name__
        self.__class__ = base
        return base.construct(self, data)
    construct.__doc__ = base.construct.__doc__
    cls.construct = construct
    for method in methods:
        msg = None
        exc = RuntimeError
        if type(method) is tuple:
            if len(method) == 2:
                method, msg = method
            else:
                method, msg, exc = method
        if not hasattr(base, method):
            raise DeveloperError('Cannot disable method %s on %s: not present on base class' % (method, cls))
        base_method = getattr(base, method)
        if type(base_method) is property:
            setattr(cls, method, _disable_property(base_method, msg, exc))
        else:
            setattr(cls, method, _disable_method(base_method, msg, exc))
    return cls