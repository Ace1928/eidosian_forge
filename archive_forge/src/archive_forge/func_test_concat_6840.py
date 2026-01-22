import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_6840():
    groupby_objs = []
    for idx, lib in enumerate((pd, pandas)):
        df1 = lib.DataFrame([['a', 1], ['b', 2], ['b', 4]], columns=['letter', 'number'])
        df1_g = df1.groupby('letter', as_index=False)['number'].agg('sum')
        df2 = lib.DataFrame([['a', 3], ['a', 4], ['b', 1]], columns=['letter', 'number'])
        df2_g = df2.groupby('letter', as_index=False)['number'].agg('sum')
        groupby_objs.append([df1_g, df2_g])
    df_equals(pd.concat(groupby_objs[0]), pandas.concat(groupby_objs[1]))