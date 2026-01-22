import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
def test_compression_binary(compression_only):
    """
    Binary file handles support compression.

    GH22555
    """
    df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    with tm.ensure_clean() as path:
        with open(path, mode='wb') as file:
            df.to_csv(file, mode='wb', compression=compression_only)
            file.seek(0)
        tm.assert_frame_equal(df, pd.read_csv(path, index_col=0, compression=compression_only))
    file = io.BytesIO()
    df.to_csv(file, mode='wb', compression=compression_only)
    file.seek(0)
    tm.assert_frame_equal(df, pd.read_csv(file, index_col=0, compression=compression_only))