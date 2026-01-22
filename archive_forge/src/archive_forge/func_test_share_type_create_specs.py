import json
from manilaclient.tests.functional.osc import base
def test_share_type_create_specs(self):
    share_type = self.create_share_type(snapshot_support=True, create_share_from_snapshot_support=True, revert_to_snapshot_support=True, mount_snapshot_support=True, extra_specs={'foo': 'bar', 'manila': 'zorilla'}, formatter='json')
    required_specs = share_type['required_extra_specs']
    optional_specs = share_type['optional_extra_specs']
    self.assertEqual(False, required_specs['driver_handles_share_servers'])
    self.assertEqual('True', optional_specs['snapshot_support'])
    self.assertEqual('True', optional_specs['create_share_from_snapshot_support'])
    self.assertEqual('True', optional_specs['revert_to_snapshot_support'])
    self.assertEqual('True', optional_specs['mount_snapshot_support'])
    self.assertEqual('bar', optional_specs['foo'])
    self.assertEqual('zorilla', optional_specs['manila'])