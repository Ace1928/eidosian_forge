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
@pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
def test_maybe_convert_numeric_infinities_raises(self, convert_to_masked_nullable):
    msg = 'Unable to parse string'
    with pytest.raises(ValueError, match=msg):
        lib.maybe_convert_numeric(np.array(['foo_inf'], dtype=object), na_values={'', 'NULL', 'nan'}, coerce_numeric=False, convert_to_masked_nullable=convert_to_masked_nullable)