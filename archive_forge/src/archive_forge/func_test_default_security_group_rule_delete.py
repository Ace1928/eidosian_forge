from unittest import mock
from unittest.mock import call
import uuid
from openstack.network.v2 import _proxy
from openstack.network.v2 import (
from openstack.test import fakes as sdk_fakes
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import default_security_group_rule
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_default_security_group_rule_delete(self):
    arglist = [self._default_sg_rules[0].id]
    verifylist = [('rule', [self._default_sg_rules[0].id])]
    self.sdk_client.find_default_security_group_rule.return_value = self._default_sg_rules[0]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.sdk_client.delete_default_security_group_rule.assert_called_once_with(self._default_sg_rules[0])
    self.assertIsNone(result)