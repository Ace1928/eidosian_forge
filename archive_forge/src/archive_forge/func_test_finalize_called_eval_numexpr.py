import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@not_implemented_mark
def test_finalize_called_eval_numexpr():
    pytest.importorskip('numexpr')
    df = pd.DataFrame({'A': [1, 2]})
    df.attrs['A'] = 1
    result = df.eval('A + 1', engine='numexpr')
    assert result.attrs == {'A': 1}