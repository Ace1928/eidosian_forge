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
def test_file_like_iterparse(xml_books, parser, mode):
    with open(xml_books, mode, encoding='utf-8' if mode == 'r' else None) as f:
        if mode == 'r' and parser == 'lxml':
            with pytest.raises(TypeError, match='reading file objects must return bytes objects'):
                read_xml(f, parser=parser, iterparse={'book': ['category', 'title', 'year', 'author', 'price']})
            return None
        else:
            df_filelike = read_xml(f, parser=parser, iterparse={'book': ['category', 'title', 'year', 'author', 'price']})
    df_expected = DataFrame({'category': ['cooking', 'children', 'web'], 'title': ['Everyday Italian', 'Harry Potter', 'Learning XML'], 'author': ['Giada De Laurentiis', 'J K. Rowling', 'Erik T. Ray'], 'year': [2005, 2005, 2003], 'price': [30.0, 29.99, 39.95]})
    tm.assert_frame_equal(df_filelike, df_expected)