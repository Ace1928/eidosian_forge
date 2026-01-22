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
def test_get_token_from_inherited_user_project_role_grants(self):
    root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
    root_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=root_id)
    leaf_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=leaf_id)
    self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)
    non_inher_up_link = self.build_role_assignment_link(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
    self.put(non_inher_up_link)
    self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(leaf_project_auth_data)
    inher_up_link = self.build_role_assignment_link(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
    self.put(inher_up_link)
    self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(leaf_project_auth_data)
    self.delete(non_inher_up_link)
    self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
    self.v3_create_token(leaf_project_auth_data)
    self.delete(inher_up_link)
    self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)