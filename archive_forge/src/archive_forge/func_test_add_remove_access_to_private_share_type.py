import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.data('2.6', '2.7')
def test_add_remove_access_to_private_share_type(self, microversion):
    self.skip_if_microversion_not_supported(microversion)
    share_type_name = data_utils.rand_name('manilaclient_functional_test')
    is_public = False
    share_type = self.create_share_type(name=share_type_name, driver_handles_share_servers='False', is_public=is_public, microversion=microversion)
    st_id = share_type['ID']
    user_project_id = self.admin_client.get_project_id(self.user_client.tenant_name)
    self._verify_access(share_type_id=st_id, is_public=is_public, microversion=microversion)
    st_access_list = self.admin_client.list_share_type_access(st_id, microversion=microversion)
    self.assertNotIn(user_project_id, st_access_list)
    self.admin_client.add_share_type_access(st_id, user_project_id, microversion=microversion)
    self.assertTrue(self._share_type_listed_by(share_type_id=st_id, by_admin=False, list_all=True))
    self.assertTrue(self._share_type_listed_by(share_type_id=st_id, by_admin=True, list_all=True))
    st_access_list = self.admin_client.list_share_type_access(st_id, microversion=microversion)
    self.assertIn(user_project_id, st_access_list)
    self.admin_client.remove_share_type_access(st_id, user_project_id, microversion=microversion)
    self._verify_access(share_type_id=st_id, is_public=is_public, microversion=microversion)
    st_access_list = self.admin_client.list_share_type_access(st_id, microversion=microversion)
    self.assertNotIn(user_project_id, st_access_list)