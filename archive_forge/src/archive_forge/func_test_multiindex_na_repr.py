from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_multiindex_na_repr(self):
    df3 = DataFrame({'A' * 30: {('A', 'A0006000', 'nuit'): 'A0006000'}, 'B' * 30: {('A', 'A0006000', 'nuit'): np.nan}, 'C' * 30: {('A', 'A0006000', 'nuit'): np.nan}, 'D' * 30: {('A', 'A0006000', 'nuit'): np.nan}, 'E' * 30: {('A', 'A0006000', 'nuit'): 'A'}, 'F' * 30: {('A', 'A0006000', 'nuit'): np.nan}})
    idf = df3.set_index(['A' * 30, 'C' * 30])
    repr(idf)