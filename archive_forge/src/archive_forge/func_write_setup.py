import io
import distutils.core
import os
import shutil
import sys
from test.support import captured_stdout
from test.support import os_helper
import unittest
from distutils.tests import support
from distutils import log
from distutils.core import setup
import os
from distutils.core import setup
from distutils.core import setup
from distutils.core import setup
from distutils.command.install import install as _install
def write_setup(self, text, path=os_helper.TESTFN):
    f = open(path, 'w')
    try:
        f.write(text)
    finally:
        f.close()
    return path