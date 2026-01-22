import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_share_group_snapshots_1234(self, **kwargs):
    sg_snapshot = {'share_group_snapshot': self.fake_share_group_snapshot}
    return (200, {}, sg_snapshot)