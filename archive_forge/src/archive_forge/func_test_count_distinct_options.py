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
def test_count_distinct_options():
    arr = pa.array([1, 2, 3, None, None])
    assert pc.count_distinct(arr).as_py() == 3
    assert pc.count_distinct(arr, mode='only_valid').as_py() == 3
    assert pc.count_distinct(arr, mode='only_null').as_py() == 1
    assert pc.count_distinct(arr, mode='all').as_py() == 4
    assert pc.count_distinct(arr, 'all').as_py() == 4