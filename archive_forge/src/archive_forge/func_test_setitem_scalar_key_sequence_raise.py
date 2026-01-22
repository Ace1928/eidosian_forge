import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='Fails to raise')
def test_setitem_scalar_key_sequence_raise(self, data):
    super().test_setitem_scalar_key_sequence_raise(data)