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
def test_round_binary():
    values = [123.456, 234.567, 345.678, 456.789, 123.456, 234.567, 345.678]
    scales = pa.array([-3, -2, -1, 0, 1, 2, 3], pa.int32())
    expected = pa.array([0, 200, 350, 457, 123.5, 234.57, 345.678], pa.float64())
    assert pc.round_binary(values, scales) == expected
    expect_zero = pa.scalar(0, pa.float64())
    expect_inf = pa.scalar(10, pa.float64())
    scale = pa.scalar(-1, pa.int32())
    assert pc.round_binary(5.0, scale, round_mode='half_towards_zero') == expect_zero
    assert pc.round_binary(5.0, scale, round_mode='half_towards_infinity') == expect_inf