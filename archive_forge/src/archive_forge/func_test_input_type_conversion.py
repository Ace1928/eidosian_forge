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
def test_input_type_conversion():
    arr = pc.add([1, 2], [4, None])
    assert arr.to_pylist() == [5, None]
    arr = pc.add([1, 2], 4)
    assert arr.to_pylist() == [5, 6]
    assert pc.equal(['foo', 'bar', None], 'foo').to_pylist() == [True, False, None]