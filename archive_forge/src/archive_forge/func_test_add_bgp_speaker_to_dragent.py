from unittest import mock
from openstack.network.v2 import bgp_speaker
from openstack.tests.unit import base
def test_add_bgp_speaker_to_dragent(self):
    sot = bgp_speaker.BgpSpeaker(**EXAMPLE)
    agent_id = '123-42'
    response = mock.Mock()
    response.status_code = 201
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    self.assertIsNone(sot.add_bgp_speaker_to_dragent(sess, agent_id))
    body = {'bgp_speaker_id': sot.id}
    url = 'agents/%s/bgp-drinstances' % agent_id
    sess.post.assert_called_with(url, json=body)