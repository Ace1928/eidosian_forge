import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_handle_overlap(self, df, df2):
    joined = merge(df, df2, on='key2', suffixes=('.foo', '.bar'))
    assert 'key1.foo' in joined
    assert 'key1.bar' in joined