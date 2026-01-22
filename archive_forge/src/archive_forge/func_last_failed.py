from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
def last_failed(self):
    suite = unittest.TestSuite()
    if not os.path.isfile(FAILED_TESTS_FNAME):
        return suite
    with open(FAILED_TESTS_FNAME) as f:
        names = f.read().split()
    for n in names:
        test = unittest.defaultTestLoader.loadTestsFromName(n)
        suite.addTest(test)
    return suite