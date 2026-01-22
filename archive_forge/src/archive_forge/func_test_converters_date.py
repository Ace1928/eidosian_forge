from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_converters_date(parser):
    convert_to_datetime = lambda x: to_datetime(x)
    df_result = read_xml(StringIO(xml_dates), converters={'date': convert_to_datetime}, parser=parser)
    df_iter = read_xml_iterparse(xml_dates, parser=parser, converters={'date': convert_to_datetime}, iterparse={'row': ['shape', 'degrees', 'sides', 'date']})
    df_expected = DataFrame({'shape': ['square', 'circle', 'triangle'], 'degrees': [360, 360, 180], 'sides': [4.0, float('nan'), 3.0], 'date': to_datetime(['2020-01-01', '2021-01-01', '2022-01-01'])})
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)