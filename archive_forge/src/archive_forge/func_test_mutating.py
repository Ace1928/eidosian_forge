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
def test_mutating(self):

    class FakeContext(object):

        def __init__(self):
            self.read_only = False

    class Fake(object):

        def __init__(self):
            self.context = FakeContext()

    def fake_function(req, context):
        return 'test passed'
    req = webob.Request.blank('/some_request')
    result = utils.mutating(fake_function)
    self.assertEqual('test passed', result(req, Fake()))