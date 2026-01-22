import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def yield_namespace_device_dtype_combinations():
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Returns
    -------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in ['numpy', 'numpy.array_api', 'cupy', 'cupy.array_api', 'torch']:
        if array_namespace == 'torch':
            for device, dtype in itertools.product(('cpu', 'cuda'), ('float64', 'float32')):
                yield (array_namespace, device, dtype)
            yield (array_namespace, 'mps', 'float32')
        else:
            yield (array_namespace, None, None)