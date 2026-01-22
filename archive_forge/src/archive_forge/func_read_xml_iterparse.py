from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def read_xml_iterparse(data, **kwargs):
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
        return read_xml(path, **kwargs)