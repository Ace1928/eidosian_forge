import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('invalid_compression', ['sfark', 'bz3', 'zipper'])
def test_invalid_compression(all_parsers, invalid_compression):
    parser = all_parsers
    compress_kwargs = {'compression': invalid_compression}
    msg = f'Unrecognized compression type: {invalid_compression}'
    with pytest.raises(ValueError, match=msg):
        parser.read_csv('test_file.zip', **compress_kwargs)