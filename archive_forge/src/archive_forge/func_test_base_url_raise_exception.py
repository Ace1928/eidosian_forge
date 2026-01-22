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
def test_base_url_raise_exception(self):
    auth = mock.Mock(spec=['get_discovery'])
    sess = mock.Mock()
    discovery = mock.Mock()
    discovery.raw_version_data = mock.Mock(return_value=[])
    auth.get_discovery = mock.Mock(return_value=discovery)
    endpoint = 'http://localhost/key_manager'
    self.assertRaises(exception.KeyManagerError, self.key_mgr._create_base_url, auth, sess, endpoint)
    auth.get_discovery.assert_called_once_with(sess, url=endpoint)
    self.assertEqual(1, discovery.raw_version_data.call_count)