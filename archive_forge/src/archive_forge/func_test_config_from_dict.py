import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_from_dict(self):
    from pecan import configuration
    conf = configuration.conf_from_dict({})
    conf['path'] = '%(confdir)s'
    self.assertTrue(os.path.samefile(conf['path'], os.getcwd()))