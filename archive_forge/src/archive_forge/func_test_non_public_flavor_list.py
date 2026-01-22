from novaclient.tests.functional import base
def test_non_public_flavor_list(self):
    flv_name = self.name_generate()
    self.nova('flavor-create --is-public false %s auto 512 1 1' % flv_name)
    self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
    flavor_list1 = self.nova('flavor-list')
    self.assertNotIn(flv_name, flavor_list1)
    flavor_list2 = self.nova('flavor-list --all')
    flavor_list3 = self.another_nova('flavor-list --all')
    self.assertIn(flv_name, flavor_list2)
    self.assertNotIn(flv_name, flavor_list3)