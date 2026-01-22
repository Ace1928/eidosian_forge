from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_replica_delete_wait(self):
    arglist = [self.share_replica.id, '--wait']
    verifylist = [('replica', [self.share_replica.id]), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.replicas_mock.delete.assert_called_with(self.share_replica, force=False)
        self.replicas_mock.get.assert_called_with(self.share_replica.id)
        self.assertIsNone(result)