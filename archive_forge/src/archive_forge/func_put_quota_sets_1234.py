import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def put_quota_sets_1234(self, *args, **kwargs):
    return (200, {}, {})