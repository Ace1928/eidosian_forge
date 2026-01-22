import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest

    This test class is used to generate tests which will run the test cases
    defined in TestGdbBindImpls in isolated subprocesses, this is for safety
    in case something goes awry.
    