import operator
import re
import numpy as np
import pytest
from pandas import option_context
import pandas._testing as tm
from pandas.core.api import (
from pandas.core.computation import expressions as expr
@pytest.mark.parametrize('fixture', ['_integer', '_integer2', '_integer_integers', '_frame', '_frame2', '_mixed', '_mixed2'])
@pytest.mark.parametrize('flex', [True, False])
@pytest.mark.parametrize('arith', ['add', 'sub', 'mul', 'mod', 'truediv', 'floordiv'])
def test_run_arithmetic(self, request, fixture, flex, arith, monkeypatch):
    df = request.getfixturevalue(fixture)
    with monkeypatch.context() as m:
        m.setattr(expr, '_MIN_ELEMENTS', 0)
        result, expected = self.call_op(df, df, flex, arith)
        if arith == 'truediv':
            assert all((x.kind == 'f' for x in expected.dtypes.values))
        tm.assert_equal(expected, result)
        for i in range(len(df.columns)):
            result, expected = self.call_op(df.iloc[:, i], df.iloc[:, i], flex, arith)
            if arith == 'truediv':
                assert expected.dtype.kind == 'f'
            tm.assert_equal(expected, result)