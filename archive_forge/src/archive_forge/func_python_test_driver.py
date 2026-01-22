import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def python_test_driver(self, tname, test_file, base_file):
    bname = os.path.basename(test_file)
    _dir = os.path.dirname(test_file)
    skip_msg = self.check_skip('test_' + tname)
    if skip_msg:
        raise _unittest.SkipTest(skip_msg)
    with open(base_file, 'r') as FILE:
        baseline = FILE.read()
    cwd = os.getcwd()
    try:
        os.chdir(_dir)
        with capture_output(None, True) as OUT:
            with LoggingIntercept(sys.stdout, formatter=pyomo_formatter):
                import_file(bname, infer_package=False, module_name='__main__')
    finally:
        os.chdir(cwd)
    try:
        self.compare_baseline(OUT.getvalue(), baseline)
    except:
        if os.environ.get('PYOMO_TEST_UPDATE_BASELINES', None):
            with open(base_file, 'w') as FILE:
                FILE.write(OUT.getvalue())
        raise