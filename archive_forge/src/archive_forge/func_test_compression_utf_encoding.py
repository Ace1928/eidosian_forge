import os
from pathlib import Path
import tarfile
import zipfile
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_compression_utf_encoding(all_parsers, csv_dir_path, utf_value, encoding_fmt):
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)
    path = os.path.join(csv_dir_path, f'utf{utf_value}_ex_small.zip')
    result = parser.read_csv(path, encoding=encoding, compression='zip', sep='\t')
    expected = DataFrame({'Country': ['Venezuela', 'Venezuela'], 'Twitter': ['Hugo Chávez Frías', 'Henrique Capriles R.']})
    tm.assert_frame_equal(result, expected)