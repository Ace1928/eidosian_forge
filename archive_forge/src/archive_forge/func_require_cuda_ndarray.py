from contextlib import contextmanager
import numpy as np
from_record_like = None
def require_cuda_ndarray(obj):
    """Raises ValueError is is_cuda_ndarray(obj) evaluates False"""
    if not is_cuda_ndarray(obj):
        raise ValueError('require an cuda ndarray object')