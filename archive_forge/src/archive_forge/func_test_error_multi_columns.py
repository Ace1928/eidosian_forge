import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('input_subset, error_message', [(list('AC'), 'columns must have matching element counts'), ([], 'column must be nonempty'), (list('AC'), 'columns must have matching element counts')])
def test_error_multi_columns(input_subset, error_message):
    df = pd.DataFrame({'A': [[0, 1, 2], np.nan, [], (3, 4)], 'B': 1, 'C': [['a', 'b', 'c'], 'foo', [], ['d', 'e', 'f']]}, index=list('abcd'))
    with pytest.raises(ValueError, match=error_message):
        df.explode(input_subset)