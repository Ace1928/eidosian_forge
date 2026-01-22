import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_bad_key(self):
    from pecan import configuration
    conf = configuration.Config({'a': 1})
    assert conf.a == 1
    self.assertRaises(AttributeError, getattr, conf, 'b')