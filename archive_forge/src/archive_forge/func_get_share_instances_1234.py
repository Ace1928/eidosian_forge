import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_share_instances_1234(self):
    return (200, {}, {'share_instance': fake_share_instance})