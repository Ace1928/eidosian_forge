from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_remove_bgp_speaker_from_dragent(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    agent_id = '123-42'
    response = mock.Mock()
    response.status_code = 204
    sess = mock.Mock()
    sess.delete = mock.Mock(return_value=response)
    self.assertIsNone(sot.remove_bgp_speaker_from_dragent(sess, agent_id))
    url = 'agents/%s/bgp-drinstances/%s' % (agent_id, IDENTIFIER)
    sess.delete.assert_called_with(url)