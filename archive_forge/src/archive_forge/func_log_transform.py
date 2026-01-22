from . import utils
from scipy import sparse
import numpy as np
import warnings
def log_transform(*args, **kwargs):
    warnings.warn('scprep.transform.log_transform is deprecated. Please use scprep.transform.log in future.', FutureWarning)
    return log(*args, **kwargs)