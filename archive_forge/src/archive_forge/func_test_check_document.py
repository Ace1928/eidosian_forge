import os
import textwrap
import unittest
from distutils.command.check import check, HAS_DOCUTILS
from distutils.tests import support
from distutils.errors import DistutilsSetupError
@unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
def test_check_document(self):
    pkg_info, dist = self.create_dist()
    cmd = check(dist)
    broken_rest = 'title\n===\n\ntest'
    msgs = cmd._check_rst_data(broken_rest)
    self.assertEqual(len(msgs), 1)
    rest = 'title\n=====\n\ntest'
    msgs = cmd._check_rst_data(rest)
    self.assertEqual(len(msgs), 0)