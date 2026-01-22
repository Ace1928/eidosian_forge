from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.slow
def test_local_file(all_parsers, csv_dir_path):
    parser = all_parsers
    kwargs = {'sep': '\t'}
    local_path = os.path.join(csv_dir_path, 'salaries.csv')
    local_result = parser.read_csv(local_path, **kwargs)
    url = 'file://localhost/' + local_path
    try:
        url_result = parser.read_csv(url, **kwargs)
        tm.assert_frame_equal(url_result, local_result)
    except URLError:
        pytest.skip('Failing on: ' + ' '.join(platform.uname()))