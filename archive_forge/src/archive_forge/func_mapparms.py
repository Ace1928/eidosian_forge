import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
def mapparms(self):
    """Return the mapping parameters.

        The returned values define a linear map ``off + scl*x`` that is
        applied to the input arguments before the series is evaluated. The
        map depends on the ``domain`` and ``window``; if the current
        ``domain`` is equal to the ``window`` the resulting map is the
        identity.  If the coefficients of the series instance are to be
        used by themselves outside this class, then the linear function
        must be substituted for the ``x`` in the standard representation of
        the base polynomials.

        Returns
        -------
        off, scl : float or complex
            The mapping function is defined by ``off + scl*x``.

        Notes
        -----
        If the current domain is the interval ``[l1, r1]`` and the window
        is ``[l2, r2]``, then the linear mapping function ``L`` is
        defined by the equations::

            L(l1) = l2
            L(r1) = r2

        """
    return pu.mapparms(self.domain, self.window)