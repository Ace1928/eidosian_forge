import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
@pytest.mark.single_cpu
def test_str_size():
    a = 'a'
    expected = sys.getsizeof(a)
    pyexe = sys.executable.replace('\\', '/')
    call = [pyexe, '-c', "a='a';import sys;sys.getsizeof(a);import pandas;print(sys.getsizeof(a));"]
    result = subprocess.check_output(call).decode()[-4:-1].strip('\n')
    assert int(result) == int(expected)