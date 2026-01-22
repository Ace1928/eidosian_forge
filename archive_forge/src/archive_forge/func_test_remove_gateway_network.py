from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_remove_gateway_network(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    response = mock.Mock()
    response.body = {'network_id': 'net_id42'}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    ret = sot.remove_gateway_network(sess, 'net_id42')
    self.assertIsNone(ret)
    body = {'network_id': 'net_id42'}
    url = 'bgp-speakers/IDENTIFIER/remove_gateway_network'
    sess.put.assert_called_with(url, json=body)