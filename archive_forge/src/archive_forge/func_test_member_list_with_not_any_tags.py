import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_member_list_with_not_any_tags(self):
    arglist = [self._mem.pool_id, '--not-any-tags', 'foo,bar']
    verifylist = [('pool', self._mem.pool_id), ('not_any_tags', ['foo', 'bar'])]
    expected_attrs = {'pool_id': self._mem.pool_id, 'not-tags-any': ['foo', 'bar']}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.member_list.assert_called_with(**expected_attrs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))