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
def test_get_stores_from_request_returns_store_from_headers(self):
    enabled_backends = {'ceph1': 'rbd', 'ceph2': 'rbd'}
    self.config(enabled_backends=enabled_backends)
    store.register_store_opts(CONF)
    self.config(default_backend='ceph1', group='glance_store')
    headers = {'x-image-meta-store': 'ceph2'}
    req = webob.Request.blank('/some_request', headers=headers)
    mp = 'glance.common.utils.glance_store.get_store_from_store_identifier'
    with mock.patch(mp) as mock_get_store:
        result = utils.get_stores_from_request(req, {})
        self.assertEqual(['ceph2'], result)
        mock_get_store.assert_called_once_with('ceph2')