from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('multi_sparse,expected', [(False, 'multiindex_sparsify_false_multi_sparse_1'), (False, 'multiindex_sparsify_false_multi_sparse_2'), (True, 'multiindex_sparsify_1'), (True, 'multiindex_sparsify_2')])
def test_to_html_multiindex_sparsify(multi_sparse, expected, datapath):
    index = MultiIndex.from_arrays([[0, 0, 1, 1], [0, 1, 0, 1]], names=['foo', None])
    df = DataFrame([[0, 1], [2, 3], [4, 5], [6, 7]], index=index)
    if expected.endswith('2'):
        df.columns = index[::2]
    with option_context('display.multi_sparse', multi_sparse):
        result = df.to_html()
    expected = expected_html(datapath, expected)
    assert result == expected