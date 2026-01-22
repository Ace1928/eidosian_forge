from unittest import mock
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import quotas as osc_quotas
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_quota_set_update_class_exception(self):
    arglist = ['default', '--class', '--gigabytes', '40']
    verifylist = [('project', 'default'), ('gigabytes', 40)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.quota_classes_mock.update.side_effect = BadRequest()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)