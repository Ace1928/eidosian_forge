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
def test_drop_null_record_batch():
    batch = pa.record_batch([pa.array(['a', None, 'c', 'd', None])], names=["a'"])
    result = batch.drop_null()
    expected = pa.record_batch([pa.array(['a', 'c', 'd'])], names=["a'"])
    assert result.equals(expected)
    batch = pa.record_batch([pa.array(['a', None, 'c', 'd', None]), pa.array([None, None, 'c', None, 'e'])], names=["a'", "b'"])
    result = batch.drop_null()
    expected = pa.record_batch([pa.array(['c']), pa.array(['c'])], names=["a'", "b'"])
    assert result.equals(expected)