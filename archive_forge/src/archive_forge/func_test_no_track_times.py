import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_no_track_times(tmp_path, setup_path):

    def checksum(filename, hash_factory=hashlib.md5, chunk_num_blocks=128):
        h = hash_factory()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b''):
                h.update(chunk)
        return h.digest()

    def create_h5_and_return_checksum(tmp_path, track_times):
        path = tmp_path / setup_path
        df = DataFrame({'a': [1]})
        with HDFStore(path, mode='w') as hdf:
            hdf.put('table', df, format='table', data_columns=True, index=None, track_times=track_times)
        return checksum(path)
    checksum_0_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_0_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)
    time.sleep(1)
    checksum_1_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_1_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)
    assert checksum_0_tt_false == checksum_1_tt_false
    assert checksum_0_tt_true != checksum_1_tt_true