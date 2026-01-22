from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
import warnings
import zipfile
from pandas._config import config
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import (
from pandas.io.excel._util import (
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer
@doc(storage_options=_shared_docs['storage_options'])
def inspect_excel_format(content_or_path: FilePath | ReadBuffer[bytes], storage_options: StorageOptions | None=None) -> str | None:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    if isinstance(content_or_path, bytes):
        content_or_path = BytesIO(content_or_path)
    with get_handle(content_or_path, 'rb', storage_options=storage_options, is_text=False) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError('stream is empty')
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)
        if any((peek.startswith(sig) for sig in XLS_SIGNATURES)):
            return 'xls'
        elif not peek.startswith(ZIP_SIGNATURE):
            return None
        with zipfile.ZipFile(stream) as zf:
            component_names = [name.replace('\\', '/').lower() for name in zf.namelist()]
        if 'xl/workbook.xml' in component_names:
            return 'xlsx'
        if 'xl/workbook.bin' in component_names:
            return 'xlsb'
        if 'content.xml' in component_names:
            return 'ods'
        return 'zip'