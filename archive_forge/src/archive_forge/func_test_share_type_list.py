import json
from manilaclient.tests.functional.osc import base
def test_share_type_list(self):
    share_type_1 = self.create_share_type(public=False)
    share_type_2 = self.create_share_type(extra_specs={'foo': 'bar'})
    types_list = self.listing_result('share type', 'list --all', client=self.admin_client)
    self.assertTableStruct(types_list, ['ID', 'Name', 'Visibility', 'Is Default', 'Required Extra Specs', 'Optional Extra Specs', 'Description'])
    id_list = [item['ID'] for item in types_list]
    self.assertIn(share_type_1['id'], id_list)
    self.assertIn(share_type_2['id'], id_list)
    types_list = self.listing_result('share type', 'list --extra-specs foo=bar')
    id_list = [item['ID'] for item in types_list]
    self.assertNotIn(share_type_1['id'], id_list)
    self.assertIn(share_type_2['id'], id_list)