import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('obj', [pd.Series([0, 0]), pd.DataFrame({'A': [0, 1], 'B': [1, 2]})])
@pytest.mark.parametrize('method', [operator.methodcaller('sum'), lambda x: x.apply(lambda y: y), lambda x: x.agg('sum'), lambda x: x.agg('mean'), lambda x: x.agg('median')])
def test_groupby_finalize(obj, method):
    obj.attrs = {'a': 1}
    result = method(obj.groupby([0, 0], group_keys=False))
    assert result.attrs == {'a': 1}