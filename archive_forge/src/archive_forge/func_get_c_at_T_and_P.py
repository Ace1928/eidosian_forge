from ._util import get_backend
from .util.pyutil import defaultnamedtuple, deprecated
from .units import default_units
def get_c_at_T_and_P(self, T, P, **kwargs):
    """Convenience method for calculating concentration

        Calculate what concentration is needed to achieve a given partial
        pressure at a specified temperature

        Parameters
        ----------
        T: float
            Temperature
        P: float
            Pressure
        \\*\\*kwargs:
            Keyword arguments passed on to :meth:`__call__`

        """
    return P * self(T, **kwargs)