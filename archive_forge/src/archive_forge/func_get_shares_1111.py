import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_shares_1111(self, **kw):
    share = {'share': {'id': 1111, 'name': 'share1111'}}
    return (200, {}, share)