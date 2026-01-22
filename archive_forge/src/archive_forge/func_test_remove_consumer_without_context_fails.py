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
def test_remove_consumer_without_context_fails(self):
    self.key_mgr._barbican_client = None
    self._test_consumer_expects_error(exception.Forbidden, self.key_mgr.remove_consumer, None, self.secret_ref)