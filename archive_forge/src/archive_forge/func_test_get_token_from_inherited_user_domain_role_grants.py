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
def test_get_token_from_inherited_user_domain_role_grants(self):
    user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    domain_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=self.domain_id)
    project_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=self.project_id)
    self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
    non_inher_ud_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=self.role_id)
    self.put(non_inher_ud_link)
    self.v3_create_token(domain_auth_data)
    self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
    inherited_role = unit.new_role_ref(name='inherited')
    PROVIDERS.role_api.create_role(inherited_role['id'], inherited_role)
    inher_ud_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=inherited_role['id'], inherited_to_projects=True)
    self.put(inher_ud_link)
    self.v3_create_token(domain_auth_data)
    self.v3_create_token(project_auth_data)
    self.delete(inher_ud_link)
    self.v3_create_token(domain_auth_data)
    self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.delete(non_inher_ud_link)
    self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)