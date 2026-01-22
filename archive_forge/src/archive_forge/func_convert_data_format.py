import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def convert_data_format(data_format, ndim):
    if data_format == 'channels_last':
        if ndim == 3:
            return 'NWC'
        elif ndim == 4:
            return 'NHWC'
        elif ndim == 5:
            return 'NDHWC'
        else:
            raise ValueError('Input rank not supported:', ndim)
    elif data_format == 'channels_first':
        if ndim == 3:
            return 'NCW'
        elif ndim == 4:
            return 'NCHW'
        elif ndim == 5:
            return 'NCDHW'
        else:
            raise ValueError('Input rank not supported:', ndim)
    else:
        raise ValueError('Invalid data_format:', data_format)