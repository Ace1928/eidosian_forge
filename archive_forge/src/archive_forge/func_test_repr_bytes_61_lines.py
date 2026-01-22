from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_bytes_61_lines(self):
    lets = list('ACDEFGHIJKLMNOP')
    slen = 50
    nseqs = 1000
    words = [[np.random.choice(lets) for x in range(slen)] for _ in range(nseqs)]
    df = DataFrame(words).astype('U1')
    assert (df.dtypes == object).all()
    repr(df)
    repr(df.iloc[:60, :])
    repr(df.iloc[:61, :])