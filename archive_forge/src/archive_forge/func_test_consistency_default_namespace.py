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
def test_consistency_default_namespace():
    pytest.importorskip('lxml')
    df_lxml = read_xml(StringIO(xml_default_nmsp), xpath='.//ns:row', namespaces={'ns': 'http://example.com'}, parser='lxml')
    df_etree = read_xml(StringIO(xml_default_nmsp), xpath='.//doc:row', namespaces={'doc': 'http://example.com'}, parser='etree')
    tm.assert_frame_equal(df_lxml, df_etree)