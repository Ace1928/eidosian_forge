from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_bgp_speaker_list(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.networkclient.bgp_speakers.assert_called_once_with(retrieve_all=True)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))