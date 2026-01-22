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
def test_list_role_assignments_for_inherited_domain_grants(self):
    """Call ``GET /role_assignments with inherited domain grants``.

        Test Plan:

        - Create 4 roles
        - Create a domain with a user and two projects
        - Assign two direct roles to project1
        - Assign a spoiler role to project2
        - Issue the URL to add inherited role to the domain
        - Issue the URL to check it is indeed on the domain
        - Issue the URL to check effective roles on project1 - this
          should return 3 roles.

        """
    role_list = []
    for _ in range(4):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    project1 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project2['id'], project2)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
    PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[1]['id'])
    PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[2]['id'])
    base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': domain['id'], 'user_id': user1['id']}
    member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
    collection_url = base_collection_url + '/inherited_to_projects'
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    r = self.get(collection_url)
    self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
    collection_url = '/role_assignments?user.id=%(user_id)s&scope.domain.id=%(domain_id)s' % {'user_id': user1['id'], 'domain_id': domain['id']}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, expected_length=1, resource_url=collection_url)
    ud_entity = self.build_role_assignment_entity(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
    self.assertRoleAssignmentInListResponse(r, ud_entity)
    collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
    r = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
    ud_url = self.build_role_assignment_link(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
    up_entity = self.build_role_assignment_entity(link=ud_url, project_id=project1['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
    self.assertRoleAssignmentInListResponse(r, up_entity)