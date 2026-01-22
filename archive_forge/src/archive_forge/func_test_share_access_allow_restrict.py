import ddt
from tempest.lib import exceptions as tempest_exc
from manilaclient.tests.functional.osc import base
@ddt.data({'lock_visibility': True, 'lock_deletion': True, 'lock_reason': None}, {'lock_visibility': False, 'lock_deletion': True, 'lock_reason': None}, {'lock_visibility': True, 'lock_deletion': False, 'lock_reason': 'testing'}, {'lock_visibility': True, 'lock_deletion': False, 'lock_reason': 'testing'})
@ddt.unpack
def test_share_access_allow_restrict(self, lock_visibility, lock_deletion, lock_reason):
    share = self.create_share()
    access_rule = self.create_share_access_rule(share=share['id'], access_type='ip', access_to='0.0.0.0/0', wait=True, lock_visibility=lock_visibility, lock_deletion=lock_deletion, lock_reason=lock_reason)
    if lock_deletion:
        self.assertRaises(tempest_exc.CommandFailed, self.openstack, 'share', params=f'access delete {share['id']} {access_rule['id']}')
    self.openstack('share', params=f'access delete {share['id']} {access_rule['id']} --unrestrict --wait')