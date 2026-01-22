import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_update_config_with_dict(self):
    from pecan import configuration
    conf = configuration.initconf()
    d = {'attr': True}
    conf['attr'] = d
    self.assertTrue(conf.attr.attr)