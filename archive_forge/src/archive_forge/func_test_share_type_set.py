import json
from manilaclient.tests.functional.osc import base
def test_share_type_set(self):
    share_type = self.create_share_type()
    self.openstack(f'share type set {share_type['id']} --description Description --name Name --public false --extra-specs foo=bar')
    share_type = json.loads(self.openstack(f'share type show {share_type['id']} -f json'))
    self.assertEqual('Description', share_type['description'])
    self.assertEqual('Name', share_type['name'])
    self.assertEqual('private', share_type['visibility'])
    self.assertEqual('bar', share_type['optional_extra_specs']['foo'])