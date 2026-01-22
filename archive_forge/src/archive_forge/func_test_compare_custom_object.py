from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_compare_custom_object(self):
    """
        Make sure non supported operations on Timedelta returns NonImplemented
        and yields to other operand (GH#20829).
        """

    class CustomClass:

        def __init__(self, cmp_result=None) -> None:
            self.cmp_result = cmp_result

        def generic_result(self):
            if self.cmp_result is None:
                return NotImplemented
            else:
                return self.cmp_result

        def __eq__(self, other):
            return self.generic_result()

        def __gt__(self, other):
            return self.generic_result()
    t = Timedelta('1s')
    assert t != 'string'
    assert t != 1
    assert t != CustomClass()
    assert t != CustomClass(cmp_result=False)
    assert t < CustomClass(cmp_result=True)
    assert not t < CustomClass(cmp_result=False)
    assert t == CustomClass(cmp_result=True)