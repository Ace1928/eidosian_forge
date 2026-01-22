import json
from manilaclient.tests.functional.osc import base
def test_share_backup_unset(self):
    share = self.create_share()
    backup = self.create_backup(share_id=share['id'], name='test_backup_unset', description='Description', backup_options={'dummy': True})
    self.openstack(f'share backup unset {backup['id']} --name --description')
    show_result = json.loads(self.openstack(f'share backup show -f json {backup['id']}'))
    self.assertEqual(backup['id'], show_result['id'])
    self.assertIsNone(show_result['name'])
    self.assertIsNone(show_result['description'])