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
def test_as_pandas(self):
    df = self.df([['a', 1.0], ['b', 2.0]], 'x:str,y:double')
    pdf = fi.as_pandas(df)
    assert [['a', 1.0], ['b', 2.0]] == pdf.values.tolist()
    df = self.df([], 'x:str,y:double')
    pdf = fi.as_pandas(df)
    assert [] == pdf.values.tolist()
    assert fi.is_local(pdf)