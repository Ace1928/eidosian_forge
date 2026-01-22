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
def test_store_with_error(self):
    self.mock_barbican.secrets.create = mock.Mock(side_effect=barbican_exceptions.HTTPClientError('test error'))
    secret_key = bytes(b'\x01\x02\xa0\xb3')
    key_length = len(secret_key) * 8
    _key = sym_key.SymmetricKey('AES', key_length, secret_key)
    self.assertRaises(exception.KeyManagerError, self.key_mgr.store, self.ctxt, _key)