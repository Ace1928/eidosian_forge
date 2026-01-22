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
def test_standardize_mapping():
    msg = 'to_dict\\(\\) only accepts initialized defaultdicts'
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(collections.defaultdict)
    msg = "unsupported type: <class 'list'>"
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping([])
    with pytest.raises(TypeError, match=msg):
        com.standardize_mapping(list)
    fill = {'bad': 'data'}
    assert com.standardize_mapping(fill) == dict
    assert com.standardize_mapping({}) == dict
    dd = collections.defaultdict(list)
    assert isinstance(com.standardize_mapping(dd), partial)