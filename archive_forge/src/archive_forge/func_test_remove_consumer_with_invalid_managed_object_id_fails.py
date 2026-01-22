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
def test_remove_consumer_with_invalid_managed_object_id_fails(self):
    side_effect = ValueError('Secret incorrectly specified.')
    self._mock_list_versions_and_test_add_consumer_expects_error(ValueError, self.ctxt, uuidutils.generate_uuid()[:-1], side_effect=side_effect)