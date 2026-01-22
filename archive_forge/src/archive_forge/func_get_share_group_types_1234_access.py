import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_group_types_1234_access(self, **kw):
    sg_type_access = {'share_group_type_access': [{'group_type_id': '11111111-1111-1111-1111-111111111111', 'project_id': '00000000-0000-0000-0000-000000000000'}]}
    return (200, {}, sg_type_access)