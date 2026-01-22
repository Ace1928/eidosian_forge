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
@pytest.mark.parametrize('val', ['', b''])
def test_empty_string_etree(val):
    with pytest.raises(ParseError, match='no element found'):
        if isinstance(val, str):
            read_xml(StringIO(val), parser='etree')
        else:
            read_xml(BytesIO(val), parser='etree')