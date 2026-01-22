import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
def test_member_show(self, mock_attrs):
    mock_attrs.return_value = {'member_id': self._mem.id, 'pool_id': self._mem.pool_id}
    arglist = [self._mem.pool_id, self._mem.id]
    verifylist = [('pool', self._mem.pool_id), ('member', self._mem.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_show.assert_called_with(member_id=self._mem.id, pool_id=self._mem.pool_id)