import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.parametrize('path_type', path_types)
def test_infer_compression_from_path(self, compression_format, path_type):
    extension, expected = compression_format
    path = path_type('foo/bar.csv' + extension)
    compression = icom.infer_compression(path, compression='infer')
    assert compression == expected