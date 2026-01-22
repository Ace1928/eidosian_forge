from __future__ import annotations
from io import (
from lzma import LZMAError
import os
from tarfile import ReadError
from urllib.error import HTTPError
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_repeat_values_new_names(parser):
    xml = '<shapes>\n  <shape>\n    <name>rectangle</name>\n    <family>rectangle</family>\n  </shape>\n  <shape>\n    <name>square</name>\n    <family>rectangle</family>\n  </shape>\n  <shape>\n    <name>ellipse</name>\n    <family>ellipse</family>\n  </shape>\n  <shape>\n    <name>circle</name>\n    <family>ellipse</family>\n  </shape>\n</shapes>'
    df_xpath = read_xml(StringIO(xml), xpath='.//shape', parser=parser, names=['name', 'group'])
    df_iter = read_xml_iterparse(xml, parser=parser, iterparse={'shape': ['name', 'family']}, names=['name', 'group'])
    df_expected = DataFrame({'name': ['rectangle', 'square', 'ellipse', 'circle'], 'group': ['rectangle', 'rectangle', 'ellipse', 'ellipse']})
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)