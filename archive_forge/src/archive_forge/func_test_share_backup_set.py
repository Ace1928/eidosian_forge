import json
from manilaclient.tests.functional.osc import base
def test_share_backup_set(self):
    share = self.create_share()
    backup = self.create_backup(share_id=share['id'], backup_options={'dummy': True})
    self.openstack(f'share backup set {backup['id']} --name test_backup_set --description Description')
    show_result = self.dict_result('share backup ', f'show {backup['id']}')
    self.assertEqual(backup['id'], show_result['id'])
    self.assertEqual('test_backup_set', show_result['name'])
    self.assertEqual('Description', show_result['description'])