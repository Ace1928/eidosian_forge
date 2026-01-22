import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
@mock.patch('octaviaclient.osc.v2.utils.get_health_monitor_attrs')
def test_health_monitor_create_wait(self, mock_client, mock_wait):
    self.hm_info['pools'] = [{'id': 'mock_pool_id'}]
    mock_client.return_value = self.hm_info
    self.api_mock.pool_show.return_value = {'loadbalancers': [{'id': 'mock_lb_id'}]}
    self.api_mock.health_monitor_show.return_value = self.hm_info
    arglist = ['mock_pool_id', '--name', self._hm.name, '--delay', str(self._hm.delay), '--timeout', str(self._hm.timeout), '--max-retries', str(self._hm.max_retries), '--type', self._hm.type.lower(), '--http-method', self._hm.http_method.lower(), '--http-version', str(self._hm.http_version), '--domain-name', self._hm.domain_name, '--wait']
    verifylist = [('pool', 'mock_pool_id'), ('name', self._hm.name), ('delay', str(self._hm.delay)), ('timeout', str(self._hm.timeout)), ('max_retries', self._hm.max_retries), ('type', self._hm.type), ('http_method', self._hm.http_method), ('http_version', self._hm.http_version), ('domain_name', self._hm.domain_name), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_create.assert_called_with(json={'healthmonitor': self.hm_info})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id='mock_lb_id', sleep_time=mock.ANY, status_field='provisioning_status')