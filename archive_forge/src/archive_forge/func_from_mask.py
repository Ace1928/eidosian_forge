import numpy as np
from .base import product
from .. import h5s, h5r, _selector
@classmethod
def from_mask(cls, mask, spaceid=None):
    """Create a point-wise selection from a NumPy boolean array """
    if not (isinstance(mask, np.ndarray) and mask.dtype.kind == 'b'):
        raise TypeError('PointSelection.from_mask only works with bool arrays')
    points = np.transpose(mask.nonzero())
    return cls(mask.shape, spaceid, points=points)