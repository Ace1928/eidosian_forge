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
def create_sample_expressions():
    schema = pa.schema([pa.field('i64', pa.int64()), pa.field('foo', pa.struct([pa.field('bar', pa.string())]))])
    a = pc.scalar(1)
    b = pc.scalar(1.1)
    c = pc.scalar(True)
    d = pc.scalar('string')
    e = pc.scalar(None)
    f = pc.scalar({'a': 1})
    g = pc.scalar(pa.scalar(1))
    h = pc.scalar(np.int64(2))
    j = pc.scalar(False)
    literal_exprs = [a, b, c, d, e, g, h, j]
    exprs_with_call = [a == b, a != b, a > b, c & j, c | j, ~c, d.is_valid(), a + b, a - b, a * b, a / b, pc.negate(a), pc.add(a, b), pc.subtract(a, b), pc.divide(a, b), pc.multiply(a, b), pc.power(a, a), pc.sqrt(a), pc.exp(b), pc.cos(b), pc.sin(b), pc.tan(b), pc.acos(b), pc.atan(b), pc.asin(b), pc.atan2(b, b), pc.abs(b), pc.sign(a), pc.bit_wise_not(a), pc.bit_wise_and(a, a), pc.bit_wise_or(a, a), pc.bit_wise_xor(a, a), pc.is_nan(b), pc.is_finite(b), pc.coalesce(a, b), a.cast(pa.int32(), safe=False)]
    exprs_with_ref = [pc.field('i64') > 5, pc.field('i64') == 5, pc.field('i64') == 7, pc.field(('foo', 'bar')) == 'value', pc.field('foo', 'bar') == 'value']
    exprs_with_numeric_refs = [pc.field(0) > 5, pc.field(0) == 5, pc.field(0) == 7, pc.field((1, 0)) == 'value', pc.field(1, 0) == 'value']
    special_cases = [f, a.isin([1, 2, 3]), pc.field('i64').is_null()]
    all_exprs = literal_exprs.copy()
    all_exprs += exprs_with_call
    all_exprs += exprs_with_ref
    all_exprs += special_cases
    return {'all': all_exprs, 'literals': literal_exprs, 'calls': exprs_with_call, 'refs': exprs_with_ref, 'numeric_refs': exprs_with_numeric_refs, 'special': special_cases, 'schema': schema}