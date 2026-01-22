from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_backups_restore
def test_pre_version(self):
    cs = fakes.FakeClient(api_version=api_versions.APIVersion('3.8'))
    b = cs.backups.get('1234')
    self.assertRaises(exc.VersionNotFoundForAPIMethod, b.update, name='new-name')