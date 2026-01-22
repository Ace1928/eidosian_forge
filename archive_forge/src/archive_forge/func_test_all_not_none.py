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
def test_all_not_none():
    assert com.all_not_none(1, 2, 3, 4)
    assert not com.all_not_none(1, 2, 3, None)
    assert not com.all_not_none(None, None, None, None)