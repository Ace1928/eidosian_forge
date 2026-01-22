from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def assert_equal_zip_safe(result: bytes, expected: bytes, compression: str):
    """
    For zip compression, only compare the CRC-32 checksum of the file contents
    to avoid checking the time-dependent last-modified timestamp which
    in some CI builds is off-by-one

    See https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    """
    if compression == 'zip':
        with zipfile.ZipFile(BytesIO(result)) as exp, zipfile.ZipFile(BytesIO(expected)) as res:
            for res_info, exp_info in zip(res.infolist(), exp.infolist()):
                assert res_info.CRC == exp_info.CRC
    elif compression == 'tar':
        with tarfile.open(fileobj=BytesIO(result)) as tar_exp, tarfile.open(fileobj=BytesIO(expected)) as tar_res:
            for tar_res_info, tar_exp_info in zip(tar_res.getmembers(), tar_exp.getmembers()):
                actual_file = tar_res.extractfile(tar_res_info)
                expected_file = tar_exp.extractfile(tar_exp_info)
                assert (actual_file is None) == (expected_file is None)
                if actual_file is not None and expected_file is not None:
                    assert actual_file.read() == expected_file.read()
    else:
        assert result == expected