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
def test_expression_call_function():
    field = pc.field('field')
    assert str(pc.hour(field)) == 'hour(field)'
    assert str(pc.round(field)) == 'round(field)'
    assert str(pc.round(field, ndigits=1)) == 'round(field, {ndigits=1, round_mode=HALF_TO_EVEN})'
    assert str(pc.add(field, 1)) == 'add(field, 1)'
    assert str(pc.add(field, pa.scalar(1))) == 'add(field, 1)'
    msg = 'only other expressions allowed as arguments'
    with pytest.raises(TypeError, match=msg):
        pc.add(field, object)