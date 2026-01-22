from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_record_export(self):
    backup_id = '76a17945-3c6f-435c-975b-b5685db10b62'
    export = cs.backups.export_record(backup_id)
    cs.assert_called('GET', '/backups/%s/export_record' % backup_id)
    self._assert_request_id(export)