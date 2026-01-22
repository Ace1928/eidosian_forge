from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_snapshot_adopt_driver_option(self):
    arglist = [self.share.id, self.export_location.fake_path, '--driver-option', 'key1=value1', '--driver-option', 'key2=value2']
    verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('driver_option', {'key1': 'value1', 'key2': 'value2'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={'key1': 'value1', 'key2': 'value2'}, name=None, description=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)