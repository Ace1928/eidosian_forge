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
def test_load_balancer_list_with_provisioning_status(self, mock_client):
    mock_client.return_value = {'provisioning_status': self._lb.provisioning_status}
    arglist = ['--provisioning-status', 'active']
    verify_list = [('provisioning_status', 'ACTIVE')]
    parsed_args = self.check_parser(self.cmd, arglist, verify_list)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))