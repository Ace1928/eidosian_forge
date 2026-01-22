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
def test_partition_nth():
    data = list(range(100, 140))
    random.shuffle(data)
    pivot = 10
    indices = pc.partition_nth_indices(data, pivot=pivot)
    check_partition_nth(data, indices, pivot, 'at_end')
    assert pc.partition_nth_indices(data, pivot) == indices
    with pytest.raises(ValueError, match="'partition_nth_indices' cannot be called without options"):
        pc.partition_nth_indices(data)