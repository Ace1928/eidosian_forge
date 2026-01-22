import cmath
import numbers
import cupy
from numpy import pi
from cupyx.scipy.fft import fft, ifft, next_fast_len
def points(self):
    """
        Return the points at which the chirp z-transform is computed.
        """
    return czt_points(self.m, self.w, self.a)