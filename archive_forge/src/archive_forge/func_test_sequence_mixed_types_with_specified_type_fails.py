import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_sequence_mixed_types_with_specified_type_fails():
    data = ['-10', '-5', {'a': 1}, '0', '5', '10']
    type = pa.string()
    with pytest.raises(TypeError):
        pa.array(data, type=type)