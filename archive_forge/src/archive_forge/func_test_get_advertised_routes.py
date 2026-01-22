from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_get_advertised_routes(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    response = mock.Mock()
    response.body = {'advertised_routes': [{'cidr': '192.168.10.0/24', 'nexthop': '10.0.0.1'}]}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=response)
    ret = sot.get_advertised_routes(sess)
    url = 'bgp-speakers/IDENTIFIER/get_advertised_routes'
    sess.get.assert_called_with(url)
    self.assertEqual(ret, response.body)