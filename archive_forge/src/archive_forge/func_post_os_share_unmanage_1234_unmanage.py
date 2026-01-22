import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_os_share_unmanage_1234_unmanage(self, **kw):
    _body = None
    resp = 202
    result = (resp, {}, _body)
    return result