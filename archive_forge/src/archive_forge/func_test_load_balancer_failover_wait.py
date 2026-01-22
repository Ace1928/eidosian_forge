import argparse
import copy
import itertools
from unittest import mock
from osc_lib import exceptions
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import load_balancer
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_status')
def test_load_balancer_failover_wait(self, mock_wait):
    arglist = [self._lb.id, '--wait']
    verifylist = [('loadbalancer', self._lb.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_failover.assert_called_with(lb_id=self._lb.id)
    mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._lb.id, sleep_time=mock.ANY, status_field='provisioning_status')