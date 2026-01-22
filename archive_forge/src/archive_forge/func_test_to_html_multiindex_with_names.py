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
def test_to_html_multiindex_with_names(self, datapath, df, expected_without_index):
    tuples = [('foo', 'car'), ('foo', 'bike'), ('bar', 'car')]
    df.index = MultiIndex.from_tuples(tuples, names=['idx1', 'idx2'])
    expected_with_index = expected_html(datapath, 'index_5')
    assert df.to_html() == expected_with_index
    assert df.to_html(index=False) == expected_without_index