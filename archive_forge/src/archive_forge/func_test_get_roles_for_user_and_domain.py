from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_roles_for_user_and_domain(self):
    """Test for getting roles for user on a domain.

        Test Plan:

        - Create a domain, with 2 users
        - Check no roles yet exit
        - Give user1 two roles on the domain, user2 one role
        - Get roles on user1 and the domain - maybe sure we only
          get back the 2 roles on user1
        - Delete both roles from user1
        - Check we get no roles back for user1 on domain

        """
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    new_user1 = unit.new_user_ref(domain_id=new_domain['id'])
    new_user1 = PROVIDERS.identity_api.create_user(new_user1)
    new_user2 = unit.new_user_ref(domain_id=new_domain['id'])
    new_user2 = PROVIDERS.identity_api.create_user(new_user2)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user1['id'], domain_id=new_domain['id'])
    self.assertEqual(0, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    PROVIDERS.assignment_api.create_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.OTHER_ROLE_ID)
    PROVIDERS.assignment_api.create_grant(user_id=new_user2['id'], domain_id=new_domain['id'], role_id=default_fixtures.ADMIN_ROLE_ID)
    roles_ids = PROVIDERS.assignment_api.get_roles_for_user_and_domain(new_user1['id'], new_domain['id'])
    self.assertEqual(2, len(roles_ids))
    self.assertIn(self.role_member['id'], roles_ids)
    self.assertIn(self.role_other['id'], roles_ids)
    PROVIDERS.assignment_api.delete_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    PROVIDERS.assignment_api.delete_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.OTHER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user1['id'], domain_id=new_domain['id'])
    self.assertEqual(0, len(roles_ref))