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
def test_filter_table():
    table = pa.table([pa.array(['a', None, 'c', 'd', 'e'])], names=['a'])
    expected_drop = pa.table([pa.array(['a', 'e'])], names=['a'])
    expected_null = pa.table([pa.array(['a', None, 'e'])], names=['a'])
    for mask in [pa.array([True, False, None, False, True]), pa.chunked_array([[True, False], [None, False, True]]), [True, False, None, False, True]]:
        result = table.filter(mask)
        assert result.equals(expected_drop)
        result = table.filter(mask, null_selection_behavior='emit_null')
        assert result.equals(expected_null)