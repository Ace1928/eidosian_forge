from numbers import Integral, Real
from .specs import (
def check_pitch(pitch):
    if not isinstance(pitch, Integral):
        raise TypeError('pichwheel value must be int')
    elif not MIN_PITCHWHEEL <= pitch <= MAX_PITCHWHEEL:
        raise ValueError('pitchwheel value must be in range {}..{}'.format(MIN_PITCHWHEEL, MAX_PITCHWHEEL))