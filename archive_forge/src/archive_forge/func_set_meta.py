import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def set_meta(self, **kwds):
    """ Update the metadata dictionary with the keywords and data provided
        here.

        Examples
        --------
        set_meta(name="Exponential", equation="y = a exp(b x) + c")
        """
    self.meta.update(kwds)