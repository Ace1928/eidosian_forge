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
def test_bad_encdoing_errors():
    with tm.ensure_clean() as path:
        with pytest.raises(LookupError, match='unknown error handler name'):
            icom.get_handle(path, 'w', errors='bad')