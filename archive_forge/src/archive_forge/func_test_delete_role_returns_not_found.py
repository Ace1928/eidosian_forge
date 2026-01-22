from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_delete_role_returns_not_found(self):
    self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.delete_role, uuid.uuid4().hex)