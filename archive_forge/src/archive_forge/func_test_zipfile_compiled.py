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
def test_zipfile_compiled(self):
    source = self.main_in_children_source
    with os_helper.temp_dir() as script_dir:
        script_name = _make_test_script(script_dir, '__main__', source=source)
        compiled_name = py_compile.compile(script_name, doraise=True)
        zip_name, run_name = make_zip_script(script_dir, 'test_zip', compiled_name)
        self._check_script(zip_name)