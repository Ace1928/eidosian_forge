from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def key_interval(self):
    """
        Returns an element in ``RealIntervalField`` which can be used as key
        for an interval tree to implement a mapping from :class:`FinitePoint`::

            sage: from sage.all import *
            sage: FinitePoint(CIF(1,2),RIF(3)).key_interval() # doctest: +NUMERIC12
            36.8919985104477?

        """
    RIF = self.z.real().parent()
    pi = RIF.pi()
    return self.z.real() + self.z.imag() * pi + self.t * pi * pi