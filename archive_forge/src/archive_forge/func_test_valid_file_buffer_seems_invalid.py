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
def test_valid_file_buffer_seems_invalid(all_parsers):

    class NoSeekTellBuffer(StringIO):

        def tell(self):
            raise AttributeError('No tell method')

        def seek(self, pos, whence=0):
            raise AttributeError('No seek method')
    data = 'a\n1'
    parser = all_parsers
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(NoSeekTellBuffer(data))
    tm.assert_frame_equal(result, expected)