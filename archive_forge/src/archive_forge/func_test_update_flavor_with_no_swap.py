from novaclient.tests.functional import base
def test_update_flavor_with_no_swap(self):
    _, flv_name = self._create_flavor()
    out = self.nova('flavor-update %s new-description' % flv_name)
    self.assertEqual(self.SWAP_DEFAULT, self._get_column_value_from_single_row_table(out, 'Swap'))