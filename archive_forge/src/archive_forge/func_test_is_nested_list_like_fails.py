import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('obj', ['abc', [], [1], (1,), ['a'], 'a', {'a'}, [1, 2, 3], Series([1]), DataFrame({'A': [1]}), ([1, 2] for _ in range(5))])
def test_is_nested_list_like_fails(obj):
    assert not inference.is_nested_list_like(obj)