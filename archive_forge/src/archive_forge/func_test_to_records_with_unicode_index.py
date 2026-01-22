from collections import abc
import email
from email.parser import Parser
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_records_with_unicode_index(self):
    result = DataFrame([{'a': 'x', 'b': 'y'}]).set_index('a').to_records()
    expected = np.rec.array([('x', 'y')], dtype=[('a', 'O'), ('b', 'O')])
    tm.assert_almost_equal(result, expected)