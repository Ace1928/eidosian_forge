import numpy as np
from .. import h5s
def read_selections_scalar(dsid, args):
    """ Returns a 2-tuple containing:

    1. Output dataset shape
    2. HDF5 dataspace containing source selection.

    Works for scalar datasets.
    """
    if dsid.shape != ():
        raise RuntimeError('Illegal selection function for non-scalar dataset')
    if args == ():
        out_shape = None
    elif args == (Ellipsis,):
        out_shape = ()
    else:
        raise ValueError('Illegal slicing argument for scalar dataspace')
    source_space = dsid.get_space()
    source_space.select_all()
    return (out_shape, source_space)