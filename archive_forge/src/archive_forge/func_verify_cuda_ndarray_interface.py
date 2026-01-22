from contextlib import contextmanager
import numpy as np
from_record_like = None
def verify_cuda_ndarray_interface(obj):
    """Verify the CUDA ndarray interface for an obj"""
    require_cuda_ndarray(obj)

    def requires_attr(attr, typ):
        if not hasattr(obj, attr):
            raise AttributeError(attr)
        if not isinstance(getattr(obj, attr), typ):
            raise AttributeError('%s must be of type %s' % (attr, typ))
    requires_attr('shape', tuple)
    requires_attr('strides', tuple)
    requires_attr('dtype', np.dtype)
    requires_attr('size', int)