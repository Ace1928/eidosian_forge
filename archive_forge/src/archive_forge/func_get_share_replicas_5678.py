import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_replicas_5678(self, **kw):
    replicas = {'share_replica': self.fake_share_replica}
    return (200, {}, replicas)