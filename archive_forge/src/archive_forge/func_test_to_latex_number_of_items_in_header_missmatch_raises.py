import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('header, num_aliases', [(['A'], 1), (('B',), 1), (('Col1', 'Col2', 'Col3'), 3), (('Col1', 'Col2', 'Col3', 'Col4'), 4)])
def test_to_latex_number_of_items_in_header_missmatch_raises(self, header, num_aliases):
    df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
    msg = f'Writing 2 cols but got {num_aliases} aliases'
    with pytest.raises(ValueError, match=msg):
        df.to_latex(header=header)