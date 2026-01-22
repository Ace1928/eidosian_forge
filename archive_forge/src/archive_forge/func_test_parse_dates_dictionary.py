from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_parse_dates_dictionary(parser):
    xml = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <shape>square</shape>\n    <degrees>360</degrees>\n    <sides>4.0</sides>\n    <year>2020</year>\n    <month>12</month>\n    <day>31</day>\n   </row>\n  <row>\n    <shape>circle</shape>\n    <degrees>360</degrees>\n    <sides/>\n    <year>2021</year>\n    <month>12</month>\n    <day>31</day>\n  </row>\n  <row>\n    <shape>triangle</shape>\n    <degrees>180</degrees>\n    <sides>3.0</sides>\n    <year>2022</year>\n    <month>12</month>\n    <day>31</day>\n  </row>\n</data>"
    df_result = read_xml(StringIO(xml), parse_dates={'date_end': ['year', 'month', 'day']}, parser=parser)
    df_iter = read_xml_iterparse(xml, parser=parser, parse_dates={'date_end': ['year', 'month', 'day']}, iterparse={'row': ['shape', 'degrees', 'sides', 'year', 'month', 'day']})
    df_expected = DataFrame({'date_end': to_datetime(['2020-12-31', '2021-12-31', '2022-12-31']), 'shape': ['square', 'circle', 'triangle'], 'degrees': [360, 360, 180], 'sides': [4.0, float('nan'), 3.0]})
    tm.assert_frame_equal(df_result, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)