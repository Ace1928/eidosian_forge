import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('func,op_name', [(lambda idx: idx - idx, '__sub__'), (lambda idx: idx + idx, '__add__'), (lambda idx: idx - ['a', 'b'], '__sub__'), (lambda idx: idx + ['a', 'b'], '__add__'), (lambda idx: ['a', 'b'] - idx, '__rsub__'), (lambda idx: ['a', 'b'] + idx, '__radd__')])
def test_disallow_addsub_ops(self, func, op_name):
    idx = Index(Categorical(['a', 'b']))
    cat_or_list = "'(Categorical|list)' and '(Categorical|list)'"
    msg = '|'.join([f'cannot perform {op_name} with this index type: CategoricalIndex', 'can only concatenate list', f'unsupported operand type\\(s\\) for [\\+-]: {cat_or_list}'])
    with pytest.raises(TypeError, match=msg):
        func(idx)