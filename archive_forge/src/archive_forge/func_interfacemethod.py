import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
def interfacemethod(func):
    """
    Convert a method specification to an actual method of the interface.

    This is a decorator that functions like `staticmethod` et al.

    The primary use of this decorator is to allow interface definitions to
    define the ``__adapt__`` method, but other interface methods can be
    overridden this way too.

    .. seealso:: `zope.interface.interfaces.IInterfaceDeclaration.interfacemethod`
    """
    f_locals = sys._getframe(1).f_locals
    methods = f_locals.setdefault(INTERFACE_METHODS, {})
    methods[func.__name__] = func
    return _decorator_non_return