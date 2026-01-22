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
def test_deep_nested_types(self):
    data = [[dict(a='1', b=[3, 4], d=1.0)], [dict(b=[30, 40])]]
    df = self.df(data, 'a:{a:str,b:[int]}')
    a = fi.as_array(df, type_safe=True)
    assert [[dict(a='1', b=[3, 4])], [dict(a=None, b=[30, 40])]] == a
    data = [[[dict(b=[30, 40])]]]
    df = self.df(data, 'a:[{a:str,b:[int]}]')
    a = fi.as_array(df, type_safe=True)
    assert [[[dict(a=None, b=[30, 40])]]] == a