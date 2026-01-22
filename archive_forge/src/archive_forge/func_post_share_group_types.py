import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_group_types(self, body, **kw):
    share_group_type = {'share_group_type': {'id': 1, 'name': 'test-group-type-1', 'share_types': body['share_group_type']['share_types'], 'is_public': True}}
    return (202, {}, share_group_type)