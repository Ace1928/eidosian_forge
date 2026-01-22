from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'check_only': False, 'restart_check': True}, {'check_only': True, 'restart_check': True}, {'check_only': True, 'restart_check': False})
@ddt.unpack
def test_share_network_subnet_create_check_api_version_exception(self, check_only, restart_check):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.69')
    arglist = [self.share_network.id]
    verifylist = [('share_network', self.share_network.id)]
    if check_only:
        arglist.append('--check-only')
        verifylist.append(('check_only', True))
    if restart_check:
        arglist.append('--restart-check')
        verifylist.append(('restart_check', True))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)