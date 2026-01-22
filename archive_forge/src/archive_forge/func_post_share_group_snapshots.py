import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_group_snapshots(self, body, **kw):
    sg_snapshot = {'share_group_snapshot': {'id': 3, 'name': 'cust_snapshot'}}
    return (202, {}, sg_snapshot)