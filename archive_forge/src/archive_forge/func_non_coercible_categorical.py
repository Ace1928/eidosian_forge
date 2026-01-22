import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.fixture
def non_coercible_categorical(monkeypatch):
    """
    Monkeypatch Categorical.__array__ to ensure no implicit conversion.

    Raises
    ------
    ValueError
        When Categorical.__array__ is called.
    """

    def array(self, dtype=None):
        raise ValueError('I cannot be converted.')
    with monkeypatch.context() as m:
        m.setattr(Categorical, '__array__', array)
        yield