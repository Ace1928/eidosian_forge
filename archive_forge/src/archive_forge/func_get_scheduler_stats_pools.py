import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_scheduler_stats_pools(self, **kw):
    pools = {'pools': [{'name': 'host1@backend1#pool1', 'host': 'host1', 'backend': 'backend1', 'pool': 'pool1'}, {'name': 'host1@backend1#pool2', 'host': 'host1', 'backend': 'backend1', 'pool': 'pool2'}]}
    return (200, {}, pools)