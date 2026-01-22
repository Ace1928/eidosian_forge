import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_servers(self, **kw):
    share_servers = {'share_servers': [{'id': 1234, 'host': 'fake_host', 'status': 'fake_status', 'share_network': 'fake_share_nw', 'project_id': 'fake_project_id', 'updated_at': 'fake_updated_at', 'name': 'fake_name', 'share_name': 'fake_share_name'}]}
    return (200, {}, share_servers)