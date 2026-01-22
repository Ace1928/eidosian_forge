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
def test_list_subclass(self):

    class MyList(list):
        pass
    val = MyList(['a'])
    assert not com.is_bool_indexer(val)
    val = MyList([True])
    assert com.is_bool_indexer(val)