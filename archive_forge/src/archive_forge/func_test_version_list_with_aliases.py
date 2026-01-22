from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def test_version_list_with_aliases(self):
    ret_data = [{'version': 'HOT123', 'type': 'hot', 'aliases': ['releasex']}, {'version': 'CFN456', 'type': 'cfn', 'aliases': ['releasey']}]
    self._stub_versions_list(ret_data)
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(['Version', 'Type', 'Aliases'], columns)
    self.assertEqual([('HOT123', 'hot', 'releasex'), ('CFN456', 'cfn', 'releasey')], list(data))