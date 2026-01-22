import os
import unittest
from apitools.gen import gen_client
from apitools.gen import test_utils
from __future__ import absolute_import
import pkgutil
def testHelp_NotEnoughArguments(self):
    with self.assertRaisesRegexp(SystemExit, '0'):
        with test_utils.CaptureOutput() as (_, err):
            gen_client.main([gen_client.__file__, '-h'])
            err_output = err.getvalue()
            self.assertIn('usage:', err_output)
            self.assertIn('error: too few arguments', err_output)