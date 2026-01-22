from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_unset_gateway_ip_qos(self):
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
    arglist = ['--external-gateway', self._network.id, '--qos-policy', qos_policy.id, '--no-qos-policy', self._router.id]
    verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('qos_policy', qos_policy.id), ('no_qos_policy', True)]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)