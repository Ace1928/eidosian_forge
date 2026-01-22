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
def test_file_handle_string_io(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2'
    fh = StringIO(data)
    parser.read_csv(fh)
    assert not fh.closed