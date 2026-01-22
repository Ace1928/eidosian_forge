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
@pytest.mark.parametrize('float_ty', [pa.float64()], ids=str)
@pytest.mark.parametrize('decimal_ty', decimal_type_traits, ids=lambda v: v.name)
@pytest.mark.parametrize('case_generator', [integral_float_to_decimal_cast_cases, real_float_to_decimal_cast_cases, random_float_to_decimal_cast_cases], ids=['integrals', 'reals', 'random'])
def test_cast_float_to_decimal(float_ty, decimal_ty, case_generator):
    with decimal.localcontext() as ctx:
        for case in case_generator(float_ty, decimal_ty.max_precision):
            check_cast_float_to_decimal(float_ty, case.float_val, decimal_ty.factory(case.precision, case.scale), ctx, decimal_ty.max_precision)