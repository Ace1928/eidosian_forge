import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_illegal_ids(self):
    from pecan import configuration
    conf = configuration.Config({})
    conf.update(configuration.conf_from_file(os.path.join(__here__, 'config_fixtures/bad/module_and_underscore.py')))
    self.assertEqual([], list(conf))