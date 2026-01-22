import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def jelly_decimal(self, d):
    """
        Jelly a decimal object.

        @param d: a decimal object to serialize.
        @type d: C{decimal.Decimal}

        @return: jelly for the decimal object.
        @rtype: C{list}
        """
    sign, guts, exponent = d.as_tuple()
    value = reduce(lambda left, right: left * 10 + right, guts)
    if sign:
        value = -value
    return [b'decimal', value, exponent]