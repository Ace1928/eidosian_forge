import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7policy_set_tag(self):
    self.api_mock.l7policy_show.return_value = {'tags': ['foo']}
    arglist = [self._l7po.id, '--tag', 'bar']
    verifylist = [('l7policy', self._l7po.id), ('tags', ['bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_set.assert_called_once()
    kwargs = self.api_mock.l7policy_set.mock_calls[0][2]
    tags = kwargs['json']['l7policy']['tags']
    self.assertEqual(2, len(tags))
    self.assertIn('foo', tags)
    self.assertIn('bar', tags)