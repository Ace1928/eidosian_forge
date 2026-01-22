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
def test_valid_hostname_fail(self):
    invalid_inputs = ['localhost.localdomain', '192.168.0.1', '☃', 'glance02.stack42.local']
    for input_str in invalid_inputs:
        self.assertFalse(utils.is_valid_hostname(input_str))