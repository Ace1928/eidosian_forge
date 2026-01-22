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
@td.skip_if_no('py.path')
def test_stringify_path_localpath(self):
    path = os.path.join('foo', 'bar')
    abs_path = os.path.abspath(path)
    lpath = LocalPath(path)
    assert icom.stringify_path(lpath) == abs_path