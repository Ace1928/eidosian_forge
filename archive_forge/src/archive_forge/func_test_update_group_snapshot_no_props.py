import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_group_snapshot_no_props(self):
    ret = cs.group_snapshots.update('1234')
    self.assertIsNone(ret)