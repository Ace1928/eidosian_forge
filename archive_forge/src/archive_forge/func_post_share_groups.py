import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_groups(self, body, **kw):
    share_group = {'share_group': {'id': 'fake-sg-id', 'name': 'fake_name'}}
    return (202, {}, share_group)