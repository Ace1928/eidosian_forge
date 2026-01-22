import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_list_with_filter(self):
    actions = self.parser.listing(self.mistral('action-list'))
    self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])
    unfiltered_len = len(actions)
    self.assertGreater(unfiltered_len, 0)
    actions = self.parser.listing(self.mistral('action-list', params='--filter name=in:std.echo,std.noop'))
    self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])
    self.assertGreater(unfiltered_len, len(actions))
    action_names = [a['Name'] for a in actions]
    self.assertIn('std.echo', action_names)
    self.assertIn('std.noop', action_names)
    self.assertNotIn('std.ssh', action_names)