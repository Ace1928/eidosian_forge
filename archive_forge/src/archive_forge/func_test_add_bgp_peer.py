from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_add_bgp_peer(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    response = mock.Mock()
    response.body = {'bgp_peer_id': '101'}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    ret = sot.add_bgp_peer(sess, '101')
    self.assertIsInstance(ret, dict)
    self.assertEqual(ret, {'bgp_peer_id': '101'})
    body = {'bgp_peer_id': '101'}
    url = 'bgp-speakers/IDENTIFIER/add_bgp_peer'
    sess.put.assert_called_with(url, json=body)