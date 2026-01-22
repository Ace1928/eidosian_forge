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
@mock.patch('octaviaclient.osc.v2.utils.get_loadbalancer_attrs')
def test_load_balancer_set(self, mock_attrs):
    qos_policy_id = uuidutils.generate_uuid()
    mock_attrs.return_value = {'loadbalancer_id': self._lb.id, 'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}
    arglist = [self._lb.id, '--name', 'new_name', '--vip-qos-policy-id', qos_policy_id]
    verifylist = [('loadbalancer', self._lb.id), ('name', 'new_name'), ('vip_qos_policy_id', qos_policy_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.load_balancer_set.assert_called_with(self._lb.id, json={'loadbalancer': {'name': 'new_name', 'vip_qos_policy_id': qos_policy_id}})