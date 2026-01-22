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
def random_float_to_decimal_cast_cases(float_ty, max_precision):
    """
    Return random-generated FloatToDecimalCase instances.
    """
    r = random.Random(42)
    for precision in range(1, max_precision, 6):
        for scale in range(0, precision, 4):
            for i in range(20):
                unscaled = r.randrange(0, 10 ** precision)
                float_val = scaled_float(unscaled, scale)
                assert float_val * 10 ** scale < 10 ** precision
                yield FloatToDecimalCase(precision, scale, float_val)