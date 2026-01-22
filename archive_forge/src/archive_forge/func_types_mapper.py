import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
def types_mapper(arrow_type):
    if pa.types.is_boolean(arrow_type):
        return pd.BooleanDtype()
    elif pa.types.is_integer(arrow_type):
        return pd.Int64Dtype()