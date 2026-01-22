import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
@classmethod
def from_tractogram(cls, tractogram):
    """Creates a :class:`LazyTractogram` object from a :class:`Tractogram` object.

        Parameters
        ----------
        tractogram : :class:`Tractgogram` object
            Tractogram from which to create a :class:`LazyTractogram` object.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
    lazy_tractogram = cls(lambda: tractogram.streamlines.copy())

    def _gen(key):
        return lambda: iter(tractogram.data_per_streamline[key])
    for k in tractogram.data_per_streamline:
        lazy_tractogram._data_per_streamline[k] = _gen(k)

    def _gen(key):
        return lambda: iter(tractogram.data_per_point[key])
    for k in tractogram.data_per_point:
        lazy_tractogram._data_per_point[k] = _gen(k)
    lazy_tractogram._nb_streamlines = len(tractogram)
    lazy_tractogram.affine_to_rasmm = tractogram.affine_to_rasmm
    return lazy_tractogram