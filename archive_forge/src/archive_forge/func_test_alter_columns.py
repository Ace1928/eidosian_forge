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
def test_alter_columns(self):
    df = self.df([], 'a:str,b:int')
    ndf = fi.alter_columns(df, 'a:str,b:str')
    assert [] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:str'
    df = self.df([['a', 1], ['c', None]], 'a:str,b:int')
    ndf = fi.alter_columns(df, 'b:int,a:str', as_fugue=True)
    assert [['a', 1], ['c', None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:int'
    df = self.df([['a', True], ['b', False], ['c', None]], 'a:str,b:bool')
    ndf = fi.alter_columns(df, 'b:str', as_fugue=True)
    actual = fi.as_array(ndf, type_safe=True)
    expected1 = [['a', 'True'], ['b', 'False'], ['c', None]]
    expected2 = [['a', 'true'], ['b', 'false'], ['c', None]]
    assert expected1 == actual or expected2 == actual
    assert fi.get_schema(ndf) == 'a:str,b:str'
    df = self.df([['a', 1], ['c', None]], 'a:str,b:int')
    ndf = fi.alter_columns(df, 'b:str', as_fugue=True)
    arr = fi.as_array(ndf, type_safe=True)
    assert [['a', '1'], ['c', None]] == arr or [['a', '1.0'], ['c', None]] == arr
    assert fi.get_schema(ndf) == 'a:str,b:str'
    df = self.df([['a', 1], ['c', None]], 'a:str,b:int')
    ndf = fi.alter_columns(df, 'b:double', as_fugue=True)
    assert [['a', 1], ['c', None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:double'
    df = self.df([['a', 1.1], ['b', None]], 'a:str,b:double')
    data = fi.as_array(fi.alter_columns(df, 'b:str', as_fugue=True), type_safe=True)
    assert [['a', '1.1'], ['b', None]] == data
    df = self.df([['a', 1.0], ['b', None]], 'a:str,b:double')
    data = fi.as_array(fi.alter_columns(df, 'b:int', as_fugue=True), type_safe=True)
    assert [['a', 1], ['b', None]] == data
    df = self.df([['a', date(2020, 1, 1)], ['b', date(2020, 1, 2)], ['c', None]], 'a:str,b:date')
    data = fi.as_array(fi.alter_columns(df, 'b:str', as_fugue=True), type_safe=True)
    assert [['a', '2020-01-01'], ['b', '2020-01-02'], ['c', None]] == data
    df = self.df([['a', datetime(2020, 1, 1, 3, 4, 5)], ['b', datetime(2020, 1, 2, 16, 7, 8)], ['c', None]], 'a:str,b:datetime')
    data = fi.as_array(fi.alter_columns(df, 'b:str', as_fugue=True), type_safe=True)
    assert [['a', '2020-01-01 03:04:05'], ['b', '2020-01-02 16:07:08'], ['c', None]] == data
    df = self.df([['a', 'trUe'], ['b', 'False'], ['c', None]], 'a:str,b:str')
    ndf = fi.alter_columns(df, 'b:bool,a:str', as_fugue=True)
    assert [['a', True], ['b', False], ['c', None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:bool'
    df = self.df([['a', '1']], 'a:str,b:str')
    ndf = fi.alter_columns(df, 'b:int,a:str')
    assert [['a', 1]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:int'
    df = self.df([['a', '1.1'], ['b', '2'], ['c', None]], 'a:str,b:str')
    ndf = fi.alter_columns(df, 'b:double', as_fugue=True)
    assert [['a', 1.1], ['b', 2.0], ['c', None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:str,b:double'
    df = self.df([['1', '2020-01-01'], ['2', '2020-01-02'], ['3', None]], 'a:str,b:str')
    ndf = fi.alter_columns(df, 'b:date,a:int', as_fugue=True)
    assert [[1, date(2020, 1, 1)], [2, date(2020, 1, 2)], [3, None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:int,b:date'
    df = self.df([['1', '2020-01-01 01:02:03'], ['2', '2020-01-02 01:02:03'], ['3', None]], 'a:str,b:str')
    ndf = fi.alter_columns(df, 'b:datetime,a:int', as_fugue=True)
    assert [[1, datetime(2020, 1, 1, 1, 2, 3)], [2, datetime(2020, 1, 2, 1, 2, 3)], [3, None]] == fi.as_array(ndf, type_safe=True)
    assert fi.get_schema(ndf) == 'a:int,b:datetime'