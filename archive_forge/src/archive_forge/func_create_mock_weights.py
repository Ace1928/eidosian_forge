import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def create_mock_weights(obj, com, adjust, ignore_na):
    if isinstance(obj, DataFrame):
        if not len(obj.columns):
            return DataFrame(index=obj.index, columns=obj.columns)
        w = concat([create_mock_series_weights(obj.iloc[:, i], com=com, adjust=adjust, ignore_na=ignore_na) for i in range(len(obj.columns))], axis=1)
        w.index = obj.index
        w.columns = obj.columns
        return w
    else:
        return create_mock_series_weights(obj, com, adjust, ignore_na)