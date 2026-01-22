import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7rule_list_with_any_tags(self):
    arglist = [self._l7po.id, '--any-tags', 'foo,bar']
    verifylist = [('l7policy', self._l7po.id), ('any_tags', ['foo', 'bar'])]
    expected_attrs = {'l7policy_id': self._l7po.id, 'tags-any': ['foo', 'bar']}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_list.assert_called_with(**expected_attrs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))