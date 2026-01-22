from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def dataframe_to_sparse(x, fill_value=0.0):
    x = pd.DataFrame.sparse.from_spmatrix(sparse.coo_matrix(x.values), index=x.index, columns=x.columns)
    x.sparse.fill_value = fill_value
    return x