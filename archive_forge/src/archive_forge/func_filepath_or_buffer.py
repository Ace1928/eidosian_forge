from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.fixture
def filepath_or_buffer(filepath_or_buffer_id, tmp_path):
    """
    A fixture yielding a string representing a filepath, a path-like object
    and a StringIO buffer. Also checks that buffer is not closed.
    """
    if filepath_or_buffer_id == 'buffer':
        buf = StringIO()
        yield buf
        assert not buf.closed
    else:
        assert isinstance(tmp_path, Path)
        if filepath_or_buffer_id == 'pathlike':
            yield (tmp_path / 'foo')
        else:
            yield str(tmp_path / 'foo')