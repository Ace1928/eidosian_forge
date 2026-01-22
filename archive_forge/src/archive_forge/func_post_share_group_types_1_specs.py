import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_share_group_types_1_specs(self, body, **kw):
    assert list(body) == ['group_specs']
    return (200, {}, {'group_specs': {'k': 'v'}})