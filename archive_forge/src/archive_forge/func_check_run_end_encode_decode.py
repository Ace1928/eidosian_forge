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
def check_run_end_encode_decode(run_end_encode_opts=None):
    arr = pa.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    encoded = pc.run_end_encode(arr, options=run_end_encode_opts)
    decoded = pc.run_end_decode(encoded)
    assert decoded.type == arr.type
    assert decoded.equals(arr)