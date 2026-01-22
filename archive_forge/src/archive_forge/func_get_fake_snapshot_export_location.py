import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_fake_snapshot_export_location():
    return {'uuid': 'foo_el_uuid', 'path': '/foo/el/path', 'share_snapshot_instance_id': 'foo_share_instance_id', 'is_admin_only': False, 'created_at': '2017-01-17T13:14:15Z', 'updated_at': '2017-01-17T14:15:16Z'}