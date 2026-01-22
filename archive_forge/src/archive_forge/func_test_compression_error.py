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
def test_compression_error(parser, compression_only):
    with tm.ensure_clean(filename='geom_xml.zip') as path:
        geom_df.to_xml(path, parser=parser, compression=compression_only)
        with pytest.raises(ParserError, match='iterparse is designed for large XML files'):
            read_xml(path, parser=parser, iterparse={'row': ['shape', 'degrees', 'sides', 'date']}, compression=compression_only)