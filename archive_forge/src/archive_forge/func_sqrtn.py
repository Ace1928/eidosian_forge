from .sage_helper import _within_sage
from .pari import *
import re
def sqrtn(self, n):
    """
        >>> r = Number(2.0, precision=100)
        >>> r.sqrtn(10)
        (1.071773462536293164213006325023, 0.809016994374947424102293417183 + 0.587785252292473129168705954639*I)
        """
    a, b = self.gen.sqrtn(n, precision=self._precision)
    return (self._parent(a), self._parent(b))