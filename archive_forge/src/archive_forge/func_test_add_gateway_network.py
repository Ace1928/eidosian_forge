from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_add_gateway_network(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    response = mock.Mock()
    response.body = {'network_id': 'net_id'}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    ret = sot.add_gateway_network(sess, 'net_id')
    self.assertIsInstance(ret, dict)
    self.assertEqual(ret, {'network_id': 'net_id'})
    body = {'network_id': 'net_id'}
    url = 'bgp-speakers/IDENTIFIER/add_gateway_network'
    sess.put.assert_called_with(url, json=body)