from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_with_multindex(self):
    index = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    data = np.zeros((8, 4))
    df = DataFrame(data, index=index)
    r = df.to_records(index=True)['level_0']
    assert 'bar' in r
    assert 'one' not in r