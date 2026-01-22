from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('append', [True, False])
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_raise_keys(self, frame_of_index_cols, drop, append):
    df = frame_of_index_cols
    with pytest.raises(KeyError, match="['foo', 'bar', 'baz']"):
        df.set_index(['foo', 'bar', 'baz'], drop=drop, append=append)
    with pytest.raises(KeyError, match='X'):
        df.set_index([df['A'], df['B'], 'X'], drop=drop, append=append)
    msg = "[('foo', 'foo', 'foo', 'bar', 'bar')]"
    with pytest.raises(KeyError, match=msg):
        df.set_index(tuple(df['A']), drop=drop, append=append)
    with pytest.raises(KeyError, match=msg):
        df.set_index(['A', df['A'], tuple(df['A'])], drop=drop, append=append)