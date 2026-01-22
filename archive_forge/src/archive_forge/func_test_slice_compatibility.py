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
def test_slice_compatibility():
    arr = pa.array(['', '𝑓', '𝑓ö', '𝑓öõ', '𝑓öõḍ', '𝑓öõḍš'])
    for start in range(-6, 6):
        for stop in itertools.chain(range(-6, 6), [None]):
            for step in [-3, -2, -1, 1, 2, 3]:
                expected = pa.array([k.as_py()[start:stop:step] for k in arr])
                result = pc.utf8_slice_codeunits(arr, start=start, stop=stop, step=step)
                assert expected.equals(result)
                assert pc.utf8_slice_codeunits(arr, start, stop, step) == result