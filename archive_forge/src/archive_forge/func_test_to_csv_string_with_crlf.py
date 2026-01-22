import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_string_with_crlf(self):
    data = {'int': [1, 2, 3], 'str_crlf': ['abc', 'd\r\nef', 'g\r\nh\r\n\r\ni']}
    df = DataFrame(data)
    with tm.ensure_clean('crlf_test.csv') as path:
        os_linesep = os.linesep.encode('utf-8')
        expected_noarg = b'int,str_crlf' + os_linesep + b'1,abc' + os_linesep + b'2,"d\r\nef"' + os_linesep + b'3,"g\r\nh\r\n\r\ni"' + os_linesep
        df.to_csv(path, index=False)
        with open(path, 'rb') as f:
            assert f.read() == expected_noarg
    with tm.ensure_clean('crlf_test.csv') as path:
        expected_lf = b'int,str_crlf\n1,abc\n2,"d\r\nef"\n3,"g\r\nh\r\n\r\ni"\n'
        df.to_csv(path, lineterminator='\n', index=False)
        with open(path, 'rb') as f:
            assert f.read() == expected_lf
    with tm.ensure_clean('crlf_test.csv') as path:
        expected_crlf = b'int,str_crlf\r\n1,abc\r\n2,"d\r\nef"\r\n3,"g\r\nh\r\n\r\ni"\r\n'
        df.to_csv(path, lineterminator='\r\n', index=False)
        with open(path, 'rb') as f:
            assert f.read() == expected_crlf