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
@pytest.mark.parametrize('suffix,archive', [('.zip', zipfile.ZipFile), ('.tar', tarfile.TarFile)])
def test_empty_archive_zip(suffix, archive):
    with tm.ensure_clean(filename=suffix) as path:
        with archive(path, 'w'):
            pass
        with pytest.raises(ValueError, match='Zero files found'):
            pd.read_csv(path)