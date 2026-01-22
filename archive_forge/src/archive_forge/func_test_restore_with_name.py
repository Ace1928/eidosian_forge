from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_backups_restore
def test_restore_with_name(self):
    cs = fakes.FakeClient(api_version=api_versions.APIVersion('3.0'))
    backup_id = '76a17945-3c6f-435c-975b-b5685db10b62'
    name = 'restore_vol'
    info = cs.restores.restore(backup_id, name=name)
    expected_body = {'restore': {'volume_id': None, 'name': name}}
    cs.assert_called('POST', '/backups/%s/restore' % backup_id, body=expected_body)
    self.assertIsInstance(info, volume_backups_restore.VolumeBackupsRestore)