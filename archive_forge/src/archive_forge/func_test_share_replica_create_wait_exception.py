from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@mock.patch('manilaclient.osc.v2.share_replicas.LOG')
def test_share_replica_create_wait_exception(self, mock_logger):
    arglist = [self.share.id, '--wait']
    verifylist = [('share', self.share.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.replicas_mock.create.assert_called_with(share=self.share, availability_zone=None)
        mock_logger.error.assert_called_with('ERROR: Share replica is in error state.')
        self.replicas_mock.get.assert_called_with(self.share_replica.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)