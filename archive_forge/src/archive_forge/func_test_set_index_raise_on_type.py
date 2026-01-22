from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('append', [True, False])
@pytest.mark.parametrize('drop', [True, False])
@pytest.mark.parametrize('box', [set], ids=['set'])
def test_set_index_raise_on_type(self, frame_of_index_cols, box, drop, append):
    df = frame_of_index_cols
    msg = 'The parameter "keys" may be a column key, .*'
    with pytest.raises(TypeError, match=msg):
        df.set_index(box(df['A']), drop=drop, append=append)
    with pytest.raises(TypeError, match=msg):
        df.set_index(['A', df['A'], box(df['A'])], drop=drop, append=append)