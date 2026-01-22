from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
@volume_shape.setter
def volume_shape(self, value):
    if value is not None:
        value = tuple(value)
        if len(value) != 3:
            raise ValueError('Volume shape should be a tuple of length 3')
        if not all((isinstance(v, int) for v in value)):
            raise ValueError('All elements of the volume shape should be integers')
    self._volume_shape = value