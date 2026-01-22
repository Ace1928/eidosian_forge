import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
@mock.patch('octaviaclient.osc.v2.utils.get_pool_attrs')
def test_pool_create_wait(self, mock_attrs, mock_wait):
    self.pool_info['loadbalancers'] = [{'id': 'mock_lb_id'}]
    mock_attrs.return_value = self.pool_info
    self.api_mock.pool_show.return_value = self.pool_info
    arglist = ['--loadbalancer', 'mock_lb_id', '--name', self._po.name, '--protocol', 'HTTP', '--lb-algorithm', 'ROUND_ROBIN', '--wait']
    verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._po.name), ('protocol', 'HTTP'), ('lb_algorithm', 'ROUND_ROBIN'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_create.assert_called_with(json={'pool': self.pool_info})
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id='mock_lb_id', sleep_time=mock.ANY, status_field='provisioning_status')