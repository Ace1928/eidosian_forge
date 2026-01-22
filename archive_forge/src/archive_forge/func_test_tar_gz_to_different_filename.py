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
def test_tar_gz_to_different_filename():
    with tm.ensure_clean(filename='.foo') as file:
        pd.DataFrame([['1', '2']], columns=['foo', 'bar']).to_csv(file, compression={'method': 'tar', 'mode': 'w:gz'}, index=False)
        with gzip.open(file) as uncompressed:
            with tarfile.TarFile(fileobj=uncompressed) as archive:
                members = archive.getmembers()
                assert len(members) == 1
                content = archive.extractfile(members[0]).read().decode('utf8')
                if is_platform_windows():
                    expected = 'foo,bar\r\n1,2\r\n'
                else:
                    expected = 'foo,bar\n1,2\n'
                assert content == expected