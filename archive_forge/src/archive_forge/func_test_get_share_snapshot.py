from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
@ddt.data(type('SnapshotUUID', (object,), {'uuid': '1234'}), type('SnapshotID', (object,), {'id': '1234'}), '1234')
def test_get_share_snapshot(self, snapshot):
    snapshot = cs.share_snapshots.get(snapshot)
    cs.assert_called('GET', '/snapshots/1234')