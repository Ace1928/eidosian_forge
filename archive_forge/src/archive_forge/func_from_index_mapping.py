from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
@classmethod
def from_index_mapping(cls, mim):
    """
        Creates a new SeriesAxis axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        SeriesAxis
        """
    start = mim.series_start * 10 ** mim.series_exponent
    step = mim.series_step * 10 ** mim.series_exponent
    return cls(start, step, mim.number_of_series_points, mim.series_unit)