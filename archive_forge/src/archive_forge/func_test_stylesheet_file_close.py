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
def test_stylesheet_file_close(kml_cta_rail_lines, xsl_flatten_doc, mode):
    pytest.importorskip('lxml')
    xsl_obj: BytesIO | StringIO
    with open(xsl_flatten_doc, mode, encoding='utf-8' if mode == 'r' else None) as f:
        if mode == 'rb':
            xsl_obj = BytesIO(f.read())
        else:
            xsl_obj = StringIO(f.read())
        read_xml(kml_cta_rail_lines, stylesheet=xsl_obj)
        assert not f.closed