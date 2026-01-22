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
def scaled_float(int_val, scale):
    """
    Return a float representation (possibly approximate) of `int_val**-scale`
    """
    assert isinstance(int_val, int)
    unscaled = decimal.Decimal(int_val)
    scaled = unscaled.scaleb(-scale)
    float_val = float(scaled)
    return float_val