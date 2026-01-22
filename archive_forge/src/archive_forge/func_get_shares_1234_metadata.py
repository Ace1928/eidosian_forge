import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_shares_1234_metadata(self, **kw):
    return (200, {}, {'metadata': {'key1': 'val1', 'key2': 'val2'}})