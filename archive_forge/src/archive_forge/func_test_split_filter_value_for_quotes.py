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
def test_split_filter_value_for_quotes(self):
    expr = '"fake\\"name",fakename,"fake,name"'
    returned = utils.split_filter_value_for_quotes(expr)
    list_values = ['fake\\"name', 'fakename', 'fake,name']
    self.assertEqual(list_values, returned)