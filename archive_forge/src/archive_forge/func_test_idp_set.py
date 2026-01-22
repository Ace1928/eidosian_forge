from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_idp_set(self):
    identity_provider = self._create_dummy_idp(add_clean_up=True)
    new_remoteid = data_utils.rand_name('newRemoteId')
    raw_output = self.openstack('identity provider set %(identity-provider)s --remote-id %(remote-id)s ' % {'identity-provider': identity_provider, 'remote-id': new_remoteid})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('identity provider show %s' % identity_provider)
    updated_value = self.parse_show_as_object(raw_output)
    self.assertIn(new_remoteid, updated_value['remote_ids'])