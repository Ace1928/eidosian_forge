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
def test_hash_aggregate_not_exported():
    for func in exported_functions:
        arrow_f = pc.get_function(func.__arrow_compute_function__['name'])
        assert arrow_f.kind != 'hash_aggregate'