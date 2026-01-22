from __future__ import annotations
from io import StringIO
import pytest
from pandas.errors import ParserWarning
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.xml import read_xml
def test_callable_func_converters(xml_books, parser, iterparse):
    with pytest.raises(TypeError, match="'float' object is not callable"):
        read_xml(xml_books, converters={'year': float()}, parser=parser, iterparse=iterparse)