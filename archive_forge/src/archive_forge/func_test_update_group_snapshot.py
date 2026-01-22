import ddt
from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_group_snapshot(self):
    s1 = cs.group_snapshots.list()[0]
    expected = {'group_snapshot': {'name': 'grp_snap2'}}
    snap = s1.update(name='grp_snap2')
    cs.assert_called('PUT', '/group_snapshots/1234', body=expected)
    self._assert_request_id(snap)
    snap = cs.group_snapshots.update('1234', name='grp_snap2')
    cs.assert_called('PUT', '/group_snapshots/1234', body=expected)
    self._assert_request_id(snap)
    snap = cs.group_snapshots.update(s1, name='grp_snap2')
    cs.assert_called('PUT', '/group_snapshots/1234', body=expected)
    self._assert_request_id(snap)