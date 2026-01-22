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
def test_match_substring():
    arr = pa.array(['ab', 'abc', 'ba', None])
    result = pc.match_substring(arr, 'ab')
    expected = pa.array([True, True, False, None])
    assert expected.equals(result)
    arr = pa.array(['áB', 'Ábc', 'ba', None])
    result = pc.match_substring(arr, 'áb', ignore_case=True)
    expected = pa.array([True, True, False, None])
    assert expected.equals(result)
    result = pc.match_substring(arr, 'áb', ignore_case=False)
    expected = pa.array([False, False, False, None])
    assert expected.equals(result)