from functools import wraps
import numpy as np
from patsy.util import (atleast_2d_column_default,
@wraps(class_)
def stateful_transform_wrapper(*args, **kwargs):
    transform = class_()
    transform.memorize_chunk(*args, **kwargs)
    transform.memorize_finish()
    return transform.transform(*args, **kwargs)