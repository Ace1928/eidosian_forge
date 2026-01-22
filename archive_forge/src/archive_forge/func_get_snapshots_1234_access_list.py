import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_snapshots_1234_access_list(self, **kw):
    access_list = {'snapshot_access_list': [{'state': 'active', 'id': '1234', 'access_type': 'ip', 'access_to': '6.6.6.6'}]}
    return (200, {}, access_list)