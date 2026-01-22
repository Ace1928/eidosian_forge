import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_barbican_endpoint_from_config(self):
    self.key_mgr.conf.barbican.barbican_endpoint = 'http://localhost:9311'
    endpoint = self.key_mgr._get_barbican_endpoint(mock.Mock(), mock.Mock())
    self.assertEqual(endpoint, 'http://localhost:9311')