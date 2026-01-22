import unittest
import sys
import os
from io import BytesIO
from distutils import cygwinccompiler
from distutils.cygwinccompiler import (check_config_h,
from distutils.tests import support
def test_check_config_h(self):
    sys.version = '2.6.1 (r261:67515, Dec  6 2008, 16:42:21) \n[GCC 4.0.1 (Apple Computer, Inc. build 5370)]'
    self.assertEqual(check_config_h()[0], CONFIG_H_OK)
    sys.version = 'something without the *CC word'
    self.assertEqual(check_config_h()[0], CONFIG_H_UNCERTAIN)
    self.write_file(self.python_h, 'xxx')
    self.assertEqual(check_config_h()[0], CONFIG_H_NOTOK)
    self.write_file(self.python_h, 'xxx __GNUC__ xxx')
    self.assertEqual(check_config_h()[0], CONFIG_H_OK)