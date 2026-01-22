import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_update_set_default_config(self):
    """Update an empty configuration with the default values"""
    from pecan import configuration
    conf = configuration.initconf()
    conf.update(configuration.conf_from_file(os.path.join(__here__, 'config_fixtures/empty.py')))
    self.assertEqual(conf.app.root, None)
    self.assertEqual(conf.app.template_path, '')
    self.assertEqual(conf.app.static_root, 'public')
    self.assertEqual(conf.server.host, '0.0.0.0')
    self.assertEqual(conf.server.port, '8080')