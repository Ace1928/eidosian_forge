from .pycode import (
from .numpy import NumPyPrinter  # NumPyPrinter is imported for backward compatibility
from sympy.core.sorting import default_sort_key
def blacklisted(self, expr):
    raise TypeError('numexpr cannot be used with %s' % expr.__class__.__name__)