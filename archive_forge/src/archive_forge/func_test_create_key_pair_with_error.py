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
def test_create_key_pair_with_error(self):
    asym_order = mock.Mock()
    self.mock_barbican.orders.create_asymmetric.return_value = asym_order
    asym_order.submit = mock.Mock(side_effect=barbican_exceptions.HTTPClientError('test error'))
    self.assertRaises(exception.KeyManagerError, self.key_mgr.create_key_pair, self.ctxt, 'RSA', 2048)