import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
def test_first_load_cached_ufunc(self):
    self.run_in_separate_process('direct_ufunc_cache_usecase()')
    self.run_in_separate_process('direct_ufunc_cache_usecase()')