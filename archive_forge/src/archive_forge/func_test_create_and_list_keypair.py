from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
from novaclient.tests.functional.v2.legacy import test_keypairs
def test_create_and_list_keypair(self):
    name = self.name_generate()
    self.nova('keypair-add %s --user %s' % (name, self.user_id))
    self.addCleanup(self.another_nova, 'keypair-delete %s' % name)
    output = self.nova('keypair-list')
    self.assertRaises(ValueError, self._get_value_from_the_table, output, name)
    output_1 = self.another_nova('keypair-list')
    output_2 = self.nova('keypair-list --user %s' % self.user_id)
    self.assertEqual(output_1, output_2)
    self.assertEqual(name, self._get_column_value_from_single_row_table(output_1, 'Name'))
    output_1 = self.another_nova('keypair-show %s ' % name)
    output_2 = self.nova('keypair-show --user %s %s' % (self.user_id, name))
    self.assertEqual(output_1, output_2)
    self.assertEqual(self.user_id, self._get_value_from_the_table(output_1, 'user_id'))