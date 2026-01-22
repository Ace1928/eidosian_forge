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
def test_wrong_compression(parser, compression, compression_only):
    actual_compression = compression
    attempted_compression = compression_only
    if actual_compression == attempted_compression:
        pytest.skip(f'{actual_compression} == {attempted_compression}')
    errors = {'bz2': (OSError, 'Invalid data stream'), 'gzip': (OSError, 'Not a gzipped file'), 'zip': (BadZipFile, 'File is not a zip file'), 'tar': (ReadError, 'file could not be opened successfully')}
    zstd = import_optional_dependency('zstandard', errors='ignore')
    if zstd is not None:
        errors['zstd'] = (zstd.ZstdError, 'Unknown frame descriptor')
    lzma = import_optional_dependency('lzma', errors='ignore')
    if lzma is not None:
        errors['xz'] = (LZMAError, 'Input format not supported by decoder')
    error_cls, error_str = errors[attempted_compression]
    with tm.ensure_clean() as path:
        geom_df.to_xml(path, parser=parser, compression=actual_compression)
        with pytest.raises(error_cls, match=error_str):
            read_xml(path, parser=parser, compression=attempted_compression)