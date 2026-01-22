import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
def test_repr_summary(self):
    with cf.option_context('display.max_seq_items', 10):
        result = repr(Index(np.arange(1000)))
        assert len(result) < 200
        assert '...' in result