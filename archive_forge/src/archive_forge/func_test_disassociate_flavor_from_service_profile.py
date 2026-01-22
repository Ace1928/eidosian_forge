from unittest import mock
from openstack.network.v2 import flavor
from openstack.tests.unit import base
def test_disassociate_flavor_from_service_profile(self):
    flav = flavor.Flavor(EXAMPLE)
    response = mock.Mock()
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    flav.id = 'IDENTIFIER'
    self.assertEqual(None, flav.disassociate_flavor_from_service_profile(sess, '1'))
    url = 'flavors/IDENTIFIER/service_profiles/1'
    sess.delete.assert_called_with(url)