import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def get_security_services_1111(self, **kw):
    ss = {'security_service': {'id': 1111, 'name': 'fake_ss'}}
    return (200, {}, ss)