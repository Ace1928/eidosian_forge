import os
import sys
import platform
import re
import gc
import operator
import warnings
from functools import partial, wraps
import shutil
import contextlib
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest
from warnings import WarningMessage
import pprint
import sysconfig
import numpy as np
from numpy.core import (
from numpy import isfinite, isnan, isinf
import numpy.linalg._umath_linalg
from io import StringIO
import unittest
def jiffies(_load_time=[]):
    """
        Return number of jiffies elapsed.

        Return number of jiffies (1/100ths of a second) that this
        process has been scheduled in user mode. See man 5 proc.

        """
    import time
    if not _load_time:
        _load_time.append(time.time())
    return int(100 * (time.time() - _load_time[0]))