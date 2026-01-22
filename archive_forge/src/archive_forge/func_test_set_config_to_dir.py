import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_set_config_to_dir(self):
    from pecan import configuration
    self.assertRaises(RuntimeError, configuration.set_config, '/')