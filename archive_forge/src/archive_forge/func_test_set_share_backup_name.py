from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.osc.v2 import share_backups as osc_share_backups
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_share_backup_name(self):
    arglist = [self.share_backup.id, '--name', 'FAKE_SHARE_BACKUP_NAME']
    verifylist = [('backup', self.share_backup.id), ('name', 'FAKE_SHARE_BACKUP_NAME')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.backups_mock.update.assert_called_with(self.share_backup, name=parsed_args.name)
    self.assertIsNone(result)