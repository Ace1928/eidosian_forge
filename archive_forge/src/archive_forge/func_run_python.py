import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *
import numpy as np
import llvmlite.binding as ll
from numba.core import utils
from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
import unittest
def run_python(args):
    p = subprocess.Popen([sys.executable] + args, cwd=self.usecase_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out, _ = p.communicate()
    rc = p.wait()
    if rc != 0:
        self.fail('python failed with the following output:\n%s' % out.decode('utf-8', 'ignore'))