from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from manilaclient.tests.functional.osc import base
def test_openstack_share_group_type_show(self):
    share_group_type_name = data_utils.rand_name('test_share_group_type_create')
    share_type_name = 'dhss_false'
    share_group_type = self.create_share_group_type(name=share_group_type_name, share_types=share_type_name)
    shares_group_type_show = self.dict_result('share', f'group type show {share_group_type_name}')
    share_type_id = self.dict_result('share', f'type show {share_type_name}')['id']
    expected_sgt_values = {'id': share_group_type['id'], 'name': share_group_type_name, 'share_types': share_type_id, 'visibility': 'public', 'is_default': 'False', 'group_specs': ''}
    for k, v in shares_group_type_show.items():
        self.assertEqual(expected_sgt_values[k], shares_group_type_show[k])