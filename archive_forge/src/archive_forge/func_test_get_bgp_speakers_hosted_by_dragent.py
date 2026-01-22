from unittest import mock
from openstack.network.v2 import agent
from openstack.tests.unit import base
def test_get_bgp_speakers_hosted_by_dragent(self):
    sot = agent.Agent(**EXAMPLE)
    sess = mock.Mock()
    response = mock.Mock()
    response.body = {'bgp_speakers': [{'name': 'bgp_speaker_1', 'ip_version': 4}]}
    response.json = mock.Mock(return_value=response.body)
    response.status_code = 200
    sess.get = mock.Mock(return_value=response)
    resp = sot.get_bgp_speakers_hosted_by_dragent(sess)
    self.assertEqual(resp, response.body)
    sess.get.assert_called_with('agents/IDENTIFIER/bgp-drinstances')