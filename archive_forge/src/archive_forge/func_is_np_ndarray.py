import numpy as np
def is_np_ndarray(value):
    return hasattr(value, '__array__') and (not (isinstance(value, np.str_) or isinstance(value, type)))