import manilaclient
from manilaclient import api_versions
from manilaclient.tests.unit.v2 import fake_clients as fakes
from manilaclient.v2 import client
def post_os_share_manage(self, body, **kw):
    _body = {'share': {'id': 'fake'}}
    resp = 202
    if not ('service_host' in body['share'] and 'share_type' in body['share'] and ('export_path' in body['share']) and ('protocol' in body['share']) and ('driver_options' in body['share'])):
        resp = 422
    result = (resp, {}, _body)
    return result