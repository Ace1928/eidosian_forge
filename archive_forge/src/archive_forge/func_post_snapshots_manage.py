import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_snapshots_manage(self, body, **kw):
    _body = {'snapshot': {'id': 'fake'}}
    resp = 202
    if not ('share_id' in body['snapshot'] and 'provider_location' in body['snapshot'] and ('driver_options' in body['snapshot'])):
        resp = 422
    result = (resp, {}, _body)
    return result