import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_ignore_compression_extension(all_parsers):
    parser = all_parsers
    df = DataFrame({'a': [0, 1]})
    with tm.ensure_clean('test.csv') as path_csv:
        with tm.ensure_clean('test.csv.zip') as path_zip:
            df.to_csv(path_csv, index=False)
            Path(path_zip).write_text(Path(path_csv).read_text(encoding='utf-8'), encoding='utf-8')
            tm.assert_frame_equal(parser.read_csv(path_zip, compression=None), df)