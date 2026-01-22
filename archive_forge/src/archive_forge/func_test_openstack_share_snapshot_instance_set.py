from manilaclient.tests.functional.osc import base
def test_openstack_share_snapshot_instance_set(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share['id'], wait=True)
    share_snapshot_instance = self.listing_result('share snapshot instance', f'list --snapshot {snapshot['id']}')
    result1 = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
    self.assertEqual(share_snapshot_instance[0]['ID'], result1['id'])
    self.assertEqual(snapshot['id'], result1['snapshot_id'])
    self.assertEqual('available', result1['status'])
    self.openstack(f'share snapshot instance set {share_snapshot_instance[0]['ID']} --status error')
    result2 = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
    self.assertEqual(share_snapshot_instance[0]['ID'], result2['id'])
    self.assertEqual('error', result2['status'])