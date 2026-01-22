import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('filename,expected_arcname', [('archive.csv', 'archive.csv'), ('archive.tsv', 'archive.tsv'), ('archive.csv.zip', 'archive.csv'), ('archive.tsv.zip', 'archive.tsv'), ('archive.zip', 'archive')])
def test_to_csv_zip_infer_name(self, tmp_path, filename, expected_arcname):
    df = DataFrame({'ABC': [1]})
    path = tmp_path / filename
    df.to_csv(path, compression='zip')
    with ZipFile(path) as zp:
        assert len(zp.filelist) == 1
        archived_file = zp.filelist[0].filename
        assert archived_file == expected_arcname