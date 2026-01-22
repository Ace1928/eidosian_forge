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
def test_remove_assignment_for_project_acting_as_domain(self):
    """Test goal: remove assignment for project acting as domain.

        Ensure when we have two role assignments for the project
        acting as domain, one dealing with it as a domain and other as a
        project, we still able to remove those assignments later.

        Test plan:
        - Create a role and a domain with a user;
        - Grant a role for this user in this domain;
        - Grant a role for this user in the same entity as a project;
        - Ensure that both assignments were created and it was valid;
        - Remove the domain assignment for the user and show that the project
        assignment for him still valid

        """
    role = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role['id'], role)
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
    assignment_domain = self.build_role_assignment_entity(role_id=role['id'], domain_id=domain['id'], user_id=user['id'], inherited_to_projects=False)
    assignment_project = self.build_role_assignment_entity(role_id=role['id'], project_id=domain['id'], user_id=user['id'], inherited_to_projects=False)
    self.put(assignment_domain['links']['assignment'])
    self.put(assignment_project['links']['assignment'])
    collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
    result = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(result, expected_length=2, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(result, assignment_domain)
    domain_url = '/domains/%s/users/%s/roles/%s' % (domain['id'], user['id'], role['id'])
    self.delete(domain_url)
    collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
    result = self.get(collection_url)
    self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
    self.assertRoleAssignmentInListResponse(result, assignment_project)