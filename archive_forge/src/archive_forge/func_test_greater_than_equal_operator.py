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
def test_greater_than_equal_operator(self):
    self.assertFalse(utils.evaluate_filter_op(9, 'gte', 10))
    self.assertTrue(utils.evaluate_filter_op(10, 'gte', 10))
    self.assertTrue(utils.evaluate_filter_op(11, 'gte', 10))