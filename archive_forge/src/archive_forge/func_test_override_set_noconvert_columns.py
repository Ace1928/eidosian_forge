from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_override_set_noconvert_columns():

    class MyTextFileReader(TextFileReader):

        def __init__(self) -> None:
            self._currow = 0
            self.squeeze = False

    class MyCParserWrapper(CParserWrapper):

        def _set_noconvert_columns(self):
            if self.usecols_dtype == 'integer':
                self.usecols = list(self.usecols)
                self.usecols.reverse()
            return CParserWrapper._set_noconvert_columns(self)
    data = 'a,b,c,d,e\n0,1,2014-01-01,09:00,4\n0,1,2014-01-02,10:00,4'
    parse_dates = [[1, 2]]
    cols = {'a': [0, 0], 'c_d': [Timestamp('2014-01-01 09:00:00'), Timestamp('2014-01-02 10:00:00')]}
    expected = DataFrame(cols, columns=['c_d', 'a'])
    parser = MyTextFileReader()
    parser.options = {'usecols': [0, 2, 3], 'parse_dates': parse_dates, 'delimiter': ','}
    parser.engine = 'c'
    parser._engine = MyCParserWrapper(StringIO(data), **parser.options)
    result = parser.read()
    tm.assert_frame_equal(result, expected)