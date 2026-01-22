from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_role_by_trustor_and_project(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    new_user = unit.new_user_ref(domain_id=new_domain['id'])
    new_user = PROVIDERS.identity_api.create_user(new_user)
    new_project = unit.new_project_ref(domain_id=new_domain['id'])
    PROVIDERS.resource_api.create_project(new_project['id'], new_project)
    role = self._create_role(domain_id=new_domain['id'])
    PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], domain_id=new_domain['id'], role_id=role['id'], inherited_to_projects=True)
    roles_ids = PROVIDERS.assignment_api.get_roles_for_trustor_and_project(new_user['id'], new_project['id'])
    self.assertEqual(2, len(roles_ids))
    self.assertIn(self.role_member['id'], roles_ids)
    self.assertIn(role['id'], roles_ids)