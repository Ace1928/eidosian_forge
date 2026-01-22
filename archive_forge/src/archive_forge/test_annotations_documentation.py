import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest

        Test some format and behavior of the html annotation with lifted loop
        