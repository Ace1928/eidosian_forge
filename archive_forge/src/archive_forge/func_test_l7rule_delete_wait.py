import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_delete')
@mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
def test_l7rule_delete_wait(self, mock_attrs, mock_wait):
    mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'l7rule_id': self._l7ru.id}
    arglist = [self._l7po.id, self._l7ru.id, '--wait']
    verifylist = [('l7policy', self._l7po.id), ('l7rule', self._l7ru.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7rule_delete.assert_called_with(l7rule_id=self._l7ru.id, l7policy_id=self._l7po.id)
    mock_wait.assert_called_once_with(manager=mock.ANY, res_id=self._l7po.id, sleep_time=mock.ANY, status_field='provisioning_status')