from . import utils
from scipy import sparse
import numpy as np
import warnings
def sqrt_transform(*args, **kwargs):
    warnings.warn('scprep.transform.sqrt_transform is deprecated. Please use scprep.transform.sqrt in future.', FutureWarning)
    return sqrt(*args, **kwargs)