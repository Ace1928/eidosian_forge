import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_is_not_set(self):
    msg = 'PECAN_CONFIG is not set and no config file was passed as an argument.'
    self.assertRaisesRegex(RuntimeError, msg, self.get_conf_path_from_env)