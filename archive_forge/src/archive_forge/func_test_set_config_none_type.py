import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_set_config_none_type(self):
    from pecan import configuration
    self.assertRaises(RuntimeError, configuration.set_config, None)