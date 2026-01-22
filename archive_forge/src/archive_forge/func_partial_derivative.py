import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def partial_derivative(self, dx, dy):
    """Construct a new spline representing a partial derivative of this
        spline.

        Parameters
        ----------
        dx, dy : int
            Orders of the derivative in x and y respectively. They must be
            non-negative integers and less than the respective degree of the
            original spline (self) in that direction (``kx``, ``ky``).

        Returns
        -------
        spline :
            A new spline of degrees (``kx - dx``, ``ky - dy``) representing the
            derivative of this spline.

        Notes
        -----

        .. versionadded:: 1.9.0

        """
    if dx == 0 and dy == 0:
        return self
    else:
        kx, ky = self.degrees
        if not (dx >= 0 and dy >= 0):
            raise ValueError('order of derivative must be positive or zero')
        if not (dx < kx and dy < ky):
            raise ValueError('order of derivative must be less than degree of spline')
        tx, ty, c = self.tck[:3]
        newc, ier = dfitpack.pardtc(tx, ty, c, kx, ky, dx, dy)
        if ier != 0:
            raise ValueError('Unexpected error code returned by pardtc: %d' % ier)
        nx = len(tx)
        ny = len(ty)
        newtx = tx[dx:nx - dx]
        newty = ty[dy:ny - dy]
        newkx, newky = (kx - dx, ky - dy)
        newclen = (nx - dx - kx - 1) * (ny - dy - ky - 1)
        return _DerivedBivariateSpline._from_tck((newtx, newty, newc[:newclen], newkx, newky))