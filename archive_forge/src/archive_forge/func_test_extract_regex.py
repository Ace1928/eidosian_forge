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
def test_extract_regex():
    ar = pa.array(['a1', 'zb2z'])
    expected = [{'letter': 'a', 'digit': '1'}, {'letter': 'b', 'digit': '2'}]
    struct = pc.extract_regex(ar, pattern='(?P<letter>[ab])(?P<digit>\\d)')
    assert struct.tolist() == expected
    struct = pc.extract_regex(ar, '(?P<letter>[ab])(?P<digit>\\d)')
    assert struct.tolist() == expected