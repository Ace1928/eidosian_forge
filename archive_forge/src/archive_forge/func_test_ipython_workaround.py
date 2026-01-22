from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import (
import multiprocess as multiprocessing
import_helper.import_module('multiprocess.synchronize')
import sys
import time
from multiprocess import Pool, set_start_method
import sys
import time
from multiprocess import Pool, set_start_method
import sys, os.path, runpy
def test_ipython_workaround(self):
    source = test_source_main_skipped_in_children
    with os_helper.temp_dir() as script_dir:
        script_name = _make_test_script(script_dir, 'ipython', source=source)
        self._check_script(script_name)
        script_no_suffix = _make_test_script(script_dir, 'ipython', source=source, omit_suffix=True)
        self._check_script(script_no_suffix)