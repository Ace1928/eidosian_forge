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
def test_module_in_package_in_zipfile(self):
    with os_helper.temp_dir() as script_dir:
        zip_name, run_name = _make_test_zip_pkg(script_dir, 'test_zip', 'test_pkg', 'script')
        launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.script', zip_name)
        self._check_script(launch_name)