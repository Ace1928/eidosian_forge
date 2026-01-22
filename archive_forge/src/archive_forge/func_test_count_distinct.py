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
def test_count_distinct():
    seed = datetime.datetime.now()
    samples = [seed.replace(year=y) for y in range(1992, 2092)]
    arr = pa.array(samples, pa.timestamp('ns'))
    assert pc.count_distinct(arr) == pa.scalar(len(samples), type=pa.int64())