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
@pytest.mark.parametrize('writer_name, writer_kwargs, module', [('to_csv', {}, 'os'), ('to_excel', {'engine': 'openpyxl'}, 'openpyxl'), ('to_feather', {}, 'pyarrow'), ('to_html', {}, 'os'), ('to_json', {}, 'os'), ('to_latex', {}, 'os'), ('to_pickle', {}, 'os'), ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')])
def test_write_fspath_all(self, writer_name, writer_kwargs, module):
    if writer_name in ['to_latex']:
        pytest.importorskip('jinja2')
    p1 = tm.ensure_clean('string')
    p2 = tm.ensure_clean('fspath')
    df = pd.DataFrame({'A': [1, 2]})
    with p1 as string, p2 as fspath:
        pytest.importorskip(module)
        mypath = CustomFSPath(fspath)
        writer = getattr(df, writer_name)
        writer(string, **writer_kwargs)
        writer(mypath, **writer_kwargs)
        with open(string, 'rb') as f_str, open(fspath, 'rb') as f_path:
            if writer_name == 'to_excel':
                result = pd.read_excel(f_str, **writer_kwargs)
                expected = pd.read_excel(f_path, **writer_kwargs)
                tm.assert_frame_equal(result, expected)
            else:
                result = f_str.read()
                expected = f_path.read()
                assert result == expected