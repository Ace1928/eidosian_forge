from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_count_substring():
    for ty, offset in [(pa.string(), pa.int32()), (pa.large_string(), pa.int64())]:
        arr = pa.array(['ab', 'cab', 'abcab', 'ba', 'AB', None], type=ty)
        result = pc.count_substring(arr, 'ab')
        expected = pa.array([1, 1, 2, 0, 0, None], type=offset)
        assert expected == result
        result = pc.count_substring(arr, 'ab', ignore_case=True)
        expected = pa.array([1, 1, 2, 0, 1, None], type=offset)
        assert expected == result