import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
@property
def smallest_normal(self):
    """Return the value for the smallest normal.

        Returns
        -------
        smallest_normal : float
            Value for the smallest normal.

        Warns
        -----
        UserWarning
            If the calculated value for the smallest normal is requested for
            double-double.
        """
    if isnan(self._machar.smallest_normal.flat[0]):
        warnings.warn('The value of smallest normal is undefined for double double', UserWarning, stacklevel=2)
    return self._machar.smallest_normal.flat[0]