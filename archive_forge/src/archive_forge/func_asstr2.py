import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def asstr2(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode('latin1')
    else:
        return str(s)