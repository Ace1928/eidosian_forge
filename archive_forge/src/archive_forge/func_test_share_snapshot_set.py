import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_set(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share=share['id'])
    self.openstack(f'share snapshot set {snapshot['id']} --name Snap --description Description')
    show_result = self.dict_result('share snapshot ', f'show {snapshot['id']}')
    self.assertEqual(snapshot['id'], show_result['id'])
    self.assertEqual('Snap', show_result['name'])
    self.assertEqual('Description', show_result['description'])