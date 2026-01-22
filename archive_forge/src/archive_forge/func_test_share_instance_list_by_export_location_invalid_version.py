from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_list_by_export_location_invalid_version(self):
    fake_export_location = '10.1.1.0:/fake_share_el'
    argslist = ['--export-location', fake_export_location]
    verifylist = [('export_location', fake_export_location)]
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.34')
    parsed_args = self.check_parser(self.cmd, argslist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)