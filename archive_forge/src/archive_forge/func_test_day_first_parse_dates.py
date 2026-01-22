from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_day_first_parse_dates(parser):
    xml = "<?xml version='1.0' encoding='utf-8'?>\n<data>\n  <row>\n    <shape>square</shape>\n    <degrees>00360</degrees>\n    <sides>4.0</sides>\n    <date>31/12/2020</date>\n   </row>\n  <row>\n    <shape>circle</shape>\n    <degrees>00360</degrees>\n    <sides/>\n    <date>31/12/2021</date>\n  </row>\n  <row>\n    <shape>triangle</shape>\n    <degrees>00180</degrees>\n    <sides>3.0</sides>\n    <date>31/12/2022</date>\n  </row>\n</data>"
    df_expected = DataFrame({'shape': ['square', 'circle', 'triangle'], 'degrees': [360, 360, 180], 'sides': [4.0, float('nan'), 3.0], 'date': to_datetime(['2020-12-31', '2021-12-31', '2022-12-31'])})
    with tm.assert_produces_warning(UserWarning, match='Parsing dates in %d/%m/%Y format'):
        df_result = read_xml(StringIO(xml), parse_dates=['date'], parser=parser)
        df_iter = read_xml_iterparse(xml, parse_dates=['date'], parser=parser, iterparse={'row': ['shape', 'degrees', 'sides', 'date']})
        tm.assert_frame_equal(df_result, df_expected)
        tm.assert_frame_equal(df_iter, df_expected)