import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_valid_fqdn(self):
    valid_inputs = ['localhost.localdomain', 'glance02.stack42.localglance04-a.stack47.local', 'img83.glance.xn--penstack-r74e.org']
    for input_str in valid_inputs:
        self.assertTrue(utils.is_valid_fqdn(input_str))