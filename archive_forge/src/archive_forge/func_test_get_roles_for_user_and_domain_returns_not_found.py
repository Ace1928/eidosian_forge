from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_roles_for_user_and_domain_returns_not_found(self):
    """Test errors raised when getting roles for user on a domain.

        Test Plan:

        - Check non-existing user gives UserNotFound
        - Check non-existing domain gives DomainNotFound

        """
    new_domain = self._get_domain_fixture()
    new_user1 = unit.new_user_ref(domain_id=new_domain['id'])
    new_user1 = PROVIDERS.identity_api.create_user(new_user1)
    self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_domain, uuid.uuid4().hex, new_domain['id'])
    self.assertRaises(exception.DomainNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_domain, new_user1['id'], uuid.uuid4().hex)