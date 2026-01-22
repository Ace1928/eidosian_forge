import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def test_health_monitor_set_wait(self, mock_wait):
    arglist = [self._hm.id, '--name', 'new_name', '--wait']
    verifylist = [('health_monitor', self._hm.id), ('name', 'new_name'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_set.assert_called_with(self._hm.id, json={'healthmonitor': {'name': 'new_name'}})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._hm.id, sleep_time=mock.ANY, status_field='provisioning_status')