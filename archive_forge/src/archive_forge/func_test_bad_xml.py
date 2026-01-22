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
def test_bad_xml(parser):
    bad_xml = "<?xml version='1.0' encoding='utf-8'?>\n  <row>\n    <shape>square</shape>\n    <degrees>00360</degrees>\n    <sides>4.0</sides>\n    <date>2020-01-01</date>\n   </row>\n  <row>\n    <shape>circle</shape>\n    <degrees>00360</degrees>\n    <sides/>\n    <date>2021-01-01</date>\n  </row>\n  <row>\n    <shape>triangle</shape>\n    <degrees>00180</degrees>\n    <sides>3.0</sides>\n    <date>2022-01-01</date>\n  </row>\n"
    with tm.ensure_clean(filename='bad.xml') as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(bad_xml)
        with pytest.raises(SyntaxError, match='Extra content at the end of the document|junk after document element'):
            read_xml(path, parser=parser, parse_dates=['date'], iterparse={'row': ['shape', 'degrees', 'sides', 'date']})