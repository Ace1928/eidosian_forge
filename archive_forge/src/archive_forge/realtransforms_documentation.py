import numpy as np
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
import functools
Forward or backward nd DCT/DST

    Parameters
    ----------
    forward : bool
        Transform direction (determines type and normalisation)
    transform : {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    