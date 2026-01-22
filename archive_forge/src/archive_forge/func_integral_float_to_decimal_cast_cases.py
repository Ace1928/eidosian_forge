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
def integral_float_to_decimal_cast_cases(float_ty, max_precision):
    """
    Return FloatToDecimalCase instances with integral values.
    """
    mantissa_digits = 16
    for precision in range(1, max_precision, 3):
        for scale in range(0, precision, 2):
            yield FloatToDecimalCase(precision, scale, 0.0)
            yield FloatToDecimalCase(precision, scale, 1.0)
            epsilon = 10 ** max(precision - mantissa_digits, scale)
            abs_maxval = largest_scaled_float_not_above(10 ** precision - epsilon, scale)
            yield FloatToDecimalCase(precision, scale, abs_maxval)