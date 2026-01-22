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
def test_s3_parser_consistency(s3_public_bucket_with_data, s3so):
    pytest.importorskip('s3fs')
    pytest.importorskip('lxml')
    s3 = f's3://{s3_public_bucket_with_data.name}/books.xml'
    df_lxml = read_xml(s3, parser='lxml', storage_options=s3so)
    df_etree = read_xml(s3, parser='etree', storage_options=s3so)
    tm.assert_frame_equal(df_lxml, df_etree)