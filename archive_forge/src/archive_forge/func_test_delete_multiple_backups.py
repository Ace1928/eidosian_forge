from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
def test_delete_multiple_backups(self):
    arglist = []
    for b in self.backups:
        arglist.append(b.id)
    verifylist = [('backups', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for b in self.backups:
        calls.append(call(b.id, ignore_missing=False, force=False))
    self.volume_sdk_client.delete_backup.assert_has_calls(calls)
    self.assertIsNone(result)