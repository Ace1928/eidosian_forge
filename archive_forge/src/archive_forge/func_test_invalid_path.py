import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_invalid_path(self):
    os.environ['PECAN_CONFIG'] = '/'
    msg = 'PECAN_CONFIG was set to an invalid path: /'
    self.assertRaisesRegex(RuntimeError, msg, self.get_conf_path_from_env)