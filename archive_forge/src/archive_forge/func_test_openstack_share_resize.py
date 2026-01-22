from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_resize(self):
    share = self.create_share()
    self.openstack(f'share resize {share['id']} 10 --wait ')
    result = self.dict_result('share', f'show {share['id']}')
    self.assertEqual('10', result['size'])