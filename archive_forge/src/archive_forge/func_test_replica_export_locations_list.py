from osc_lib import utils as oscutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_replica_export_locations_list(self):
    arglist = [self.share_replica.id]
    verifylist = [('replica', self.share_replica.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.replicas_mock.get.assert_called_with(self.share_replica.id)
    self.export_locations_mock.list.assert_called_with(self.share_replica)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.values, data)