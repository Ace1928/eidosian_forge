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
def test_share_snapshot_create_metadata(self):
    arglist = [self.share.id, '--name', self.share_snapshot.name, '--description', self.share_snapshot.description, '--property', 'Manila=zorilla', '--property', 'Zorilla=manila']
    verifylist = [('share', self.share.id), ('name', self.share_snapshot.name), ('description', self.share_snapshot.description), ('property', {'Manila': 'zorilla', 'Zorilla': 'manila'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=self.share_snapshot.name, description=self.share_snapshot.description, metadata={'Manila': 'zorilla', 'Zorilla': 'manila'})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)