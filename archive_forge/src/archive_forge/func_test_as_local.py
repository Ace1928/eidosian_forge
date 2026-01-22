from datetime import date, datetime
from typing import Any
from unittest import TestCase
import numpy as np
import pandas as pd
from pytest import raises
import fugue.api as fi
from fugue.dataframe import ArrowDataFrame, DataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
def test_as_local(self):
    with raises(NotImplementedError):
        fi.as_local(10)
    with raises(NotImplementedError):
        fi.as_local_bounded(10)
    df = self.df([['a', 1.0], ['b', 2.0]], 'x:str,y:double')
    ldf = fi.as_local(df)
    assert fi.is_local(ldf)
    lbdf = fi.as_local_bounded(df)
    assert fi.is_local(lbdf) and fi.is_bounded(lbdf)
    fdf = fi.as_fugue_df(df)
    fdf.reset_metadata({'a': 1})
    ldf = fi.as_local(fdf)
    assert ldf.metadata == {'a': 1}
    lbdf = fi.as_local_bounded(fdf)
    assert fi.is_local(lbdf) and fi.is_bounded(lbdf)
    assert ldf.metadata == {'a': 1}