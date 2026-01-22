import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_snapshot_instances(self, **kw):
    instances = {'snapshot_instances': [self.fake_snapshot_instance]}
    return (200, {}, instances)