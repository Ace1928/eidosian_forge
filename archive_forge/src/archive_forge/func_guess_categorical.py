import numpy as np
import six
from patsy import PatsyError
from patsy.util import (SortAnythingKey,
def guess_categorical(data):
    if safe_is_pandas_categorical(data):
        return True
    if isinstance(data, _CategoricalBox):
        return True
    data = np.asarray(data)
    if safe_issubdtype(data.dtype, np.number):
        return False
    return True