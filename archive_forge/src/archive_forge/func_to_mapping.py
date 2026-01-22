from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
def to_mapping(self, dim):
    """
        Converts the SeriesAxis to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`cifti2.Cifti2MatrixIndicesMap`
        """
    mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_SERIES')
    mim.series_exponent = 0
    mim.series_start = self.start
    mim.series_step = self.step
    mim.number_of_series_points = self.size
    mim.series_unit = self.unit
    return mim