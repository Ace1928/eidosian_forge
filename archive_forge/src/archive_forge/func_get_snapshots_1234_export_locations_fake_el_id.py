import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_snapshots_1234_export_locations_fake_el_id(self, **kw):
    return (200, {}, {'share_snapshot_export_location': {'id': 'fake_id', 'path': '/fake_path'}})