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
def test_repr_html_long(self):
    with option_context('display.max_rows', 60):
        max_rows = get_option('display.max_rows')
        h = max_rows - 1
        df = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
        reg_repr = df._repr_html_()
        assert '..' not in reg_repr
        assert str(41 + max_rows // 2) in reg_repr
        h = max_rows + 1
        df = DataFrame({'A': np.arange(1, 1 + h), 'B': np.arange(41, 41 + h)})
        long_repr = df._repr_html_()
        assert '..' in long_repr
        assert str(41 + max_rows // 2) not in long_repr
        assert f'{h} rows ' in long_repr
        assert '2 columns' in long_repr