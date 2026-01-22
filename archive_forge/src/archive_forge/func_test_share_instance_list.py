from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_list(self):
    argslist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, argslist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertIs(True, self.instances_mock.list.called)
    self.assertEqual(self.column_headers, columns)
    self.assertEqual(list(self.instance_values), list(data))