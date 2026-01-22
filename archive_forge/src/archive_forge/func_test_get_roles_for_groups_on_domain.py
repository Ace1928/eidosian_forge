from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_roles_for_groups_on_domain(self):
    """Test retrieving group domain roles.

        Test Plan:

        - Create a domain, three groups and three roles
        - Assign one an inherited and the others a non-inherited group role
          to the domain
        - Ensure that only the non-inherited roles are returned on the domain

        """
    domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
    group_list = []
    group_id_list = []
    role_list = []
    for _ in range(3):
        group = unit.new_group_ref(domain_id=domain1['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_list.append(group)
        group_id_list.append(group['id'])
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], domain_id=domain1['id'], role_id=role_list[2]['id'], inherited_to_projects=True)
    role_refs = PROVIDERS.assignment_api.get_roles_for_groups(group_id_list, domain_id=domain1['id'])
    self.assertThat(role_refs, matchers.HasLength(2))
    self.assertIn(role_list[0], role_refs)
    self.assertIn(role_list[1], role_refs)