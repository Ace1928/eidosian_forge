from novaclient.tests.functional import base
def test_add_access_non_public_flavor(self):
    flv_name = self.name_generate()
    self.nova('flavor-create --is-public false %s auto 512 1 1' % flv_name)
    self.addCleanup(self.nova, 'flavor-delete %s' % flv_name)
    self.nova('flavor-access-add', params='%s %s' % (flv_name, self.project_id))
    self.assertIn(self.project_id, self.nova('flavor-access-list --flavor %s' % flv_name))