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
def getTaggedValue(self, tag):
    """ Returns the value associated with 'tag'. """
    value = self.queryTaggedValue(tag, default=_marker)
    if value is _marker:
        raise KeyError(tag)
    return value