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
def test_empty_attrs_only(parser):
    xml = '\n      <data>\n        <row>\n          <shape sides="4">square</shape>\n          <degrees>360</degrees>\n        </row>\n        <row>\n          <shape sides="0">circle</shape>\n          <degrees>360</degrees>\n        </row>\n        <row>\n          <shape sides="3">triangle</shape>\n          <degrees>180</degrees>\n        </row>\n      </data>'
    with pytest.raises(ValueError, match='xpath does not return any nodes or attributes'):
        read_xml(StringIO(xml), xpath='./row', attrs_only=True, parser=parser)