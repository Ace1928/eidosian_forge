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
def test_repeat_names(parser):
    xml = '<shapes>\n  <shape type="2D">\n    <name>circle</name>\n    <type>curved</type>\n  </shape>\n  <shape type="3D">\n    <name>sphere</name>\n    <type>curved</type>\n  </shape>\n</shapes>'
    df_xpath = read_xml(StringIO(xml), xpath='.//shape', parser=parser, names=['type_dim', 'shape', 'type_edge'])
    df_iter = read_xml_iterparse(xml, parser=parser, iterparse={'shape': ['type', 'name', 'type']}, names=['type_dim', 'shape', 'type_edge'])
    df_expected = DataFrame({'type_dim': ['2D', '3D'], 'shape': ['circle', 'sphere'], 'type_edge': ['curved', 'curved']})
    tm.assert_frame_equal(df_xpath, df_expected)
    tm.assert_frame_equal(df_iter, df_expected)