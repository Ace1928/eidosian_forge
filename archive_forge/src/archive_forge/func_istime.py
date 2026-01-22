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
def istime(x):
    return x.dtype.char in 'Mm'