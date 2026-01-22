import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
def test_member_create_with_tag(self, mock_attrs):
    mock_attrs.return_value = {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'pool_id': self._mem.pool_id, 'tags': ['foo']}
    arglist = ['pool_id', '--address', '192.0.2.122', '--protocol-port', '80', '--tag', 'foo']
    verifylist = [('address', '192.0.2.122'), ('protocol_port', 80), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.member_create.assert_called_with(pool_id=self._mem.pool_id, json={'member': {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'tags': ['foo']}})