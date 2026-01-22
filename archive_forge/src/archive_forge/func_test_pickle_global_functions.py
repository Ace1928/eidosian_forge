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
def test_pickle_global_functions(pickle_module):
    for name in pc.list_functions():
        try:
            func = getattr(pc, name)
        except AttributeError:
            continue
        reconstructed = pickle_module.loads(pickle_module.dumps(func))
        assert reconstructed is func