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
def test_mixed_sequence_errors():
    with pytest.raises(ValueError, match='tried to convert to boolean'):
        pa.array([True, 'foo'], type=pa.bool_())
    with pytest.raises(ValueError, match='tried to convert to float32'):
        pa.array([1.5, 'foo'], type=pa.float32())
    with pytest.raises(ValueError, match='tried to convert to double'):
        pa.array([1.5, 'foo'])