import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
def test_l7rule_set_tag(self, mock_attrs):
    self.api_mock.l7rule_show.return_value = {'tags': ['foo']}
    mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id, 'tags': ['bar']}
    arglist = [self._l7po.id, self._l7ru.id, '--tag', 'bar']
    verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('tags', ['bar'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_set.assert_called_once()
    kwargs = self.api_mock.l7rule_set.mock_calls[0][2]
    tags = kwargs['json']['rule']['tags']
    self.assertEqual(2, len(tags))
    self.assertIn('foo', tags)
    self.assertIn('bar', tags)