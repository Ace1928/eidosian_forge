from castellan.common.credentials import keystone_password
from castellan.tests import base
def test___ne___user_domain_name(self):
    other_user_domain_name = 'domain0'
    other_ks_password_credential = keystone_password.KeystonePassword(self.password, username=self.username, user_id=self.user_id, user_domain_id=self.domain_id, user_domain_name=other_user_domain_name, trust_id=self.trust_id, domain_id=self.domain_id, domain_name=self.domain_name, project_id=self.project_id, project_name=self.project_name, project_domain_id=self.project_domain_id, project_domain_name=self.project_domain_name, reauthenticate=self.reauthenticate)
    self.assertTrue(self.ks_password_credential != other_ks_password_credential)