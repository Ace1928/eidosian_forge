import cupy
from cupyx.scipy.interpolate._interpolate import PPoly
@classmethod
def from_spline(cls, tck, extrapolate=None):
    raise NotImplementedError('This method does not make sense for an Akima interpolator.')