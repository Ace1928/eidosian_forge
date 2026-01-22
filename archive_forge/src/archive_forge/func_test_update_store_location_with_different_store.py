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
def test_update_store_location_with_different_store(self):
    enabled_backends = {'ceph1': 'rbd', 'ceph2': 'rbd'}
    self.config(enabled_backends=enabled_backends)
    self._test_update_store_in_location({'store': 'rbd2'}, 'ceph1', 'ceph1')