from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_list_by_export_location(self):
    fake_export_location = '10.1.1.0:/fake_share_el'
    argslist = ['--export-location', fake_export_location]
    verifylist = [('export_location', fake_export_location)]
    parsed_args = self.check_parser(self.cmd, argslist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.instances_mock.list.assert_called_with(export_location=fake_export_location)
    self.assertEqual(self.column_headers, columns)
    self.assertEqual(list(self.instance_values), list(data))