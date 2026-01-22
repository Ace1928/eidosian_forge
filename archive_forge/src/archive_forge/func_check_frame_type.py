from numbers import Integral, Real
from .specs import (
def check_frame_type(value):
    if not isinstance(value, Integral):
        raise TypeError('frame_type must be int')
    elif not 0 <= value <= 7:
        raise ValueError('frame_type must be in range 0..7')