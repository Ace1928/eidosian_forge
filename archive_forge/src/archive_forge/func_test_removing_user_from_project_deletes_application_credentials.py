import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_removing_user_from_project_deletes_application_credentials(self):
    app_cred_proj_A_1 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app1')
    app_cred_proj_A_2 = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name='app2')
    app_cred_proj_B = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_baz['id'], name='app3')
    PROVIDERS.assignment_api.add_role_to_user_and_project(project_id=self.project_baz['id'], user_id=self.user_foo['id'], role_id=self.role__member_['id'])
    self.app_cred_api.create_application_credential(app_cred_proj_A_1)
    self.app_cred_api.create_application_credential(app_cred_proj_A_2)
    self.app_cred_api.create_application_credential(app_cred_proj_B)
    self.assertIn(app_cred_proj_A_1['id'], self._list_ids(self.user_foo))
    self.assertIn(app_cred_proj_A_2['id'], self._list_ids(self.user_foo))
    self.assertIn(app_cred_proj_B['id'], self._list_ids(self.user_foo))
    self.app_cred_api.get_application_credential(app_cred_proj_A_1['id'])
    self.app_cred_api.get_application_credential(app_cred_proj_A_2['id'])
    self.app_cred_api.get_application_credential(app_cred_proj_B['id'])
    PROVIDERS.assignment_api.remove_role_from_user_and_project(user_id=self.user_foo['id'], project_id=self.project_bar['id'], role_id=self.role__member_['id'])
    self.assertNotIn(app_cred_proj_A_1['id'], self._list_ids(self.user_foo))
    self.assertNotIn(app_cred_proj_A_2['id'], self._list_ids(self.user_foo))
    self.assertIn(app_cred_proj_B['id'], self._list_ids(self.user_foo))
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, app_cred_proj_A_1['id'])
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, app_cred_proj_A_2['id'])
    self.assertEqual(app_cred_proj_B['id'], self.app_cred_api.get_application_credential(app_cred_proj_B['id'])['id'])