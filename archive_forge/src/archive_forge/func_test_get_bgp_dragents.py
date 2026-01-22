from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_get_bgp_dragents(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    response = mock.Mock()
    response.body = {'agents': [{'binary': 'neutron-bgp-dragent', 'alive': True}]}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess = mock.Mock()
    sess.get = mock.Mock(return_value=response)
    ret = sot.get_bgp_dragents(sess)
    url = 'bgp-speakers/IDENTIFIER/bgp-dragents'
    sess.get.assert_called_with(url)
    self.assertEqual(ret, response.body)