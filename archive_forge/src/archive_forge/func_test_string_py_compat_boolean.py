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
@pytest.mark.parametrize('function_name', ['is_alnum', 'is_alpha', 'is_ascii', 'is_decimal', 'is_digit', 'is_lower', 'is_numeric', 'is_printable', 'is_space', 'is_upper'])
@pytest.mark.parametrize('variant', ['ascii', 'utf8'])
def test_string_py_compat_boolean(function_name, variant):
    arrow_name = variant + '_' + function_name
    py_name = function_name.replace('_', '')
    ignore = codepoints_ignore.get(function_name, set()) | find_new_unicode_codepoints()
    for i in range(128 if ascii else 69632):
        if i in range(55296, 57344):
            continue
        if i in ignore:
            continue
        c = chr(i)
        if hasattr(pc, arrow_name) and function_name != 'is_space':
            ar = pa.array([c])
            arrow_func = getattr(pc, arrow_name)
            assert arrow_func(ar)[0].as_py() == getattr(c, py_name)()