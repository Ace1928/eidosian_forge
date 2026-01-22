import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('method', [operator.methodcaller('add_categories', ['c']), operator.methodcaller('as_ordered'), operator.methodcaller('as_unordered'), lambda x: getattr(x, 'codes'), operator.methodcaller('remove_categories', 'a'), operator.methodcaller('remove_unused_categories'), operator.methodcaller('rename_categories', {'a': 'A', 'b': 'B'}), operator.methodcaller('reorder_categories', ['b', 'a']), operator.methodcaller('set_categories', ['A', 'B'])])
@not_implemented_mark
def test_categorical_accessor(method):
    s = pd.Series(['a', 'b'], dtype='category')
    s.attrs = {'a': 1}
    result = method(s.cat)
    assert result.attrs == {'a': 1}