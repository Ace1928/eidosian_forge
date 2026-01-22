from numbers import Integral, Real
from .specs import (
def check_data_byte(value):
    if not isinstance(value, Integral):
        raise TypeError('data byte must be int')
    elif not 0 <= value <= 127:
        raise ValueError('data byte must be in range 0..127')