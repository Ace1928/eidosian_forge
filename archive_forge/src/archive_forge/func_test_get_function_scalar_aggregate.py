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
def test_get_function_scalar_aggregate():
    _check_get_function('mean', pc.ScalarAggregateFunction, pc.ScalarAggregateKernel, 8)