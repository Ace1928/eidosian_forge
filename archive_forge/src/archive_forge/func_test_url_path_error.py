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
@pytest.mark.network
@pytest.mark.single_cpu
def test_url_path_error(parser, httpserver, xml_file):
    with open(xml_file, encoding='utf-8') as f:
        httpserver.serve_content(content=f.read())
        with pytest.raises(ParserError, match='iterparse is designed for large XML files'):
            read_xml(httpserver.url, parser=parser, iterparse={'row': ['shape', 'degrees', 'sides', 'date']})