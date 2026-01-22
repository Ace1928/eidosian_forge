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
def test_compression_warning(compression_only):
    df = pd.DataFrame(100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]], columns=['X', 'Y', 'Z'])
    with tm.ensure_clean() as path:
        with icom.get_handle(path, 'w', compression=compression_only) as handles:
            with tm.assert_produces_warning(RuntimeWarning):
                df.to_csv(handles.handle, compression=compression_only)