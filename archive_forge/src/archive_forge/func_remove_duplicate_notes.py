from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from . import (evaluation_io, MultiClassEvaluation, SumEvaluation,
from .onsets import onset_evaluation, OnsetEvaluation
from ..io import load_notes
def remove_duplicate_notes(data):
    """
    Remove duplicate rows from the array.

    Parameters
    ----------
    data : numpy array
        Data.

    Returns
    -------
    numpy array
        Data array with duplicate rows removed.

    Notes
    -----
    This function removes only exact duplicates.

    """
    if data.size == 0:
        return data
    order = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
    unique = np.unique(order, return_index=True)[1]
    data = data[unique]
    return data[data[:, 0].argsort()]