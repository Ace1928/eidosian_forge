import numpy as np
from .. import h5s
def select_read(fspace, args):
    """ Top-level dispatch function for reading.

    At the moment, only supports reading from scalar datasets.
    """
    if fspace.shape == ():
        return ScalarReadSelection(fspace, args)
    raise NotImplementedError()