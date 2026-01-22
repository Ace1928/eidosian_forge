import modin.numpy as np
import modin.pandas as pd
def try_convert_from_interoperable_type(obj, copy=False):
    if isinstance(obj, _INTEROPERABLE_TYPES):
        obj = np.array(obj, copy=copy)
    return obj