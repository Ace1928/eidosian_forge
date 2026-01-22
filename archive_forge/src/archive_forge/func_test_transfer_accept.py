from manilaclient.tests.functional.osc import base
def test_transfer_accept(self):
    share = self.create_share(share_type=self.share_type['name'], wait_for_status='available', client=self.user_client)
    transfer = self.create_share_transfer(share['id'], name='transfer_test')
    self._wait_for_object_status('share', share['id'], 'awaiting_transfer')
    self.openstack(f'share transfer accept {transfer['id']} {transfer['auth_key']}')
    self._wait_for_object_status('share', share['id'], 'available')