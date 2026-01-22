import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_writes_tar_gz(all_parsers):
    parser = all_parsers
    data = DataFrame({'Country': ['Venezuela', 'Venezuela'], 'Twitter': ['Hugo Chávez Frías', 'Henrique Capriles R.']})
    with tm.ensure_clean('test.tar.gz') as tar_path:
        data.to_csv(tar_path, index=False)
        tm.assert_frame_equal(parser.read_csv(tar_path), data)
        with tarfile.open(tar_path, 'r:gz') as tar:
            result = parser.read_csv(tar.extractfile(tar.getnames()[0]), compression='infer')
            tm.assert_frame_equal(result, data)