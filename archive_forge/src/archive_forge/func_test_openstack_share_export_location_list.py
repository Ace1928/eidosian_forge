from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_export_location_list(self):
    share = self.create_share()
    share_export_locations = self.get_share_export_locations(share['id'])
    result_export_locations = self.listing_result('share', f'export location list {share['id']}')
    self.assertTableStruct(result_export_locations, ['ID', 'Path'])
    export_location_ids = [el['ID'] for el in share_export_locations]
    for share_export in result_export_locations:
        self.assertIn(share_export['ID'], export_location_ids)