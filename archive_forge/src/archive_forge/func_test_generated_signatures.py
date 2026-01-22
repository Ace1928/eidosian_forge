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
def test_generated_signatures():
    sig = inspect.signature(pc.add)
    assert str(sig) == '(x, y, /, *, memory_pool=None)'
    sig = inspect.signature(pc.min_max)
    assert str(sig) == '(array, /, *, skip_nulls=True, min_count=1, options=None, memory_pool=None)'
    sig = inspect.signature(pc.quantile)
    assert str(sig) == "(array, /, q=0.5, *, interpolation='linear', skip_nulls=True, min_count=0, options=None, memory_pool=None)"
    sig = inspect.signature(pc.binary_join_element_wise)
    assert str(sig) == "(*strings, null_handling='emit_null', null_replacement='', options=None, memory_pool=None)"
    sig = inspect.signature(pc.choose)
    assert str(sig) == '(indices, /, *values, memory_pool=None)'
    sig = inspect.signature(pc.random)
    assert str(sig) == "(n, *, initializer='system', options=None, memory_pool=None)"