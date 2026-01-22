import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_filtered_role_assignments_for_inherited_grants(self):
    """Call ``GET /role_assignments?scope.OS-INHERIT:inherited_to``.

        Test Plan:

        - Create 5 roles
        - Create a domain with a user, group and two projects
        - Assign three direct spoiler roles to projects
        - Issue the URL to add an inherited user role to the domain
        - Issue the URL to add an inherited group role to the domain
        - Issue the URL to filter by inherited roles - this should
          return just the 2 inherited roles.

        """
    role_list = []
    for _ in range(5):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    group1 = unit.new_group_ref(domain_id=domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    project1 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
    PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[1]['id'])
    PROVIDERS.assignment_api.create_grant(role_list[2]['id'], user_id=user1['id'], domain_id=domain['id'])
    base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': domain['id'], 'user_id': user1['id']}
    member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
    collection_url = base_collection_url + '/inherited_to_projects'
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    r = self.get(collection_url)
    self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
    base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': domain['id'], 'group_id': group1['id']}
    member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[4]['id']}
    collection_url = base_collection_url + '/inherited_to_projects'
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    r = self.get(collection_url)
    self.assertValidRoleListResponse(r, ref=role_list[4], resource_url=collection_url)
    collection_url = '/role_assignments?scope.OS-INHERIT:inherited_to=projects'
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
    ud_entity = self.build_role_assignment_entity(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
    gd_entity = self.build_role_assignment_entity(domain_id=domain['id'], group_id=group1['id'], role_id=role_list[4]['id'], inherited_to_projects=True)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    self.assertRoleAssignmentInListResponse(r, gd_entity)