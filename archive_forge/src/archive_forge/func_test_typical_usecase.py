import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_typical_usecase():
    df = pd.DataFrame([{'var1': 'a,b,c', 'var2': 1}, {'var1': 'd,e,f', 'var2': 2}], columns=['var1', 'var2'])
    exploded = df.var1.str.split(',').explode()
    result = df[['var2']].join(exploded)
    expected = pd.DataFrame({'var2': [1, 1, 1, 2, 2, 2], 'var1': list('abcdef')}, columns=['var2', 'var1'], index=[0, 0, 0, 1, 1, 1])
    tm.assert_frame_equal(result, expected)