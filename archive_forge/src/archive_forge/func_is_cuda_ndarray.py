from contextlib import contextmanager
import numpy as np
from_record_like = None
def is_cuda_ndarray(obj):
    """Check if an object is a CUDA ndarray"""
    return getattr(obj, '__cuda_ndarray__', False)