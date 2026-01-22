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
@pytest.mark.parametrize('method', ['to_string', 'to_html', 'to_latex'])
def test_filepath_or_buffer_bad_arg_raises(float_frame, method):
    if method in ['to_latex']:
        pytest.importorskip('jinja2')
    msg = 'buf is not a file name and it has no write method'
    with pytest.raises(TypeError, match=msg):
        getattr(float_frame, method)(buf=object())