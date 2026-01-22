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
def test_replace_plain():
    data = pa.array(['foozfoo', 'food', None])
    ar = pc.replace_substring(data, pattern='foo', replacement='bar')
    assert ar.tolist() == ['barzbar', 'bard', None]
    ar = pc.replace_substring(data, 'foo', 'bar')
    assert ar.tolist() == ['barzbar', 'bard', None]
    ar = pc.replace_substring(data, pattern='foo', replacement='bar', max_replacements=1)
    assert ar.tolist() == ['barzfoo', 'bard', None]
    ar = pc.replace_substring(data, 'foo', 'bar', max_replacements=1)
    assert ar.tolist() == ['barzfoo', 'bard', None]