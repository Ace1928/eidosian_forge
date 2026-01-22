from manilaclient.tests.functional.osc import base
def test_openstack_share_snapshot_instance_list(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share['id'])
    share_snapshot_instances_list = self.listing_result('share snapshot instance', 'list --detailed')
    self.assertTableStruct(share_snapshot_instances_list, ['ID', 'Snapshot ID', 'Status', 'Created At', 'Updated At', 'Share ID', 'Share Instance ID', 'Progress', 'Provider Location'])
    self.assertIn(snapshot['id'], [item['Snapshot ID'] for item in share_snapshot_instances_list])