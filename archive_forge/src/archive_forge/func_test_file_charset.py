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
def test_file_charset(xml_doc_ch_utf, parser):
    df_file = read_xml(xml_doc_ch_utf, parser=parser)
    df_expected = DataFrame({'問': ['問  若箇是邪而言破邪 何者是正而道(Sorry, this is Big5 only)申正', '問 既破有得申無得 亦應但破性執申假名以不', '問 既破性申假 亦應但破有申無 若有無兩洗 亦應性假雙破耶'], '答': [''.join(['答  邪既無量 正亦多途  大略為言不出二種 謂', '有得與無得 有得是邪須破 無得是正須申\n\t\t故']), None, '答  不例  有無皆是性 所以須雙破 既分性假異 故有破不破'], 'a': [None, '答 性執是有得 假名是無得  今破有得申無得 即是破性執申假名也', None]})
    tm.assert_frame_equal(df_file, df_expected)