import contextlib
import os
import numpy as np
from .cudadrv import devicearray, devices, driver
from numba.core import config
from numba.cuda.api_util import prepare_shape_strides_dtype
@require_context
def legacy_default_stream():
    """
    Get the legacy default CUDA stream.
    """
    return current_context().get_legacy_default_stream()