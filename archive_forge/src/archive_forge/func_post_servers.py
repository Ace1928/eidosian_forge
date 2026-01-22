from novaclient import api_versions
from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
from novaclient.tests.unit.v2 import fakes as v2_fakes
def post_servers(self, request, context):
    body = request.json()
    context.status_code = 202
    assert set(body.keys()) <= set(['server', 'os:scheduler_hints'])
    fakes.assert_has_keys(body['server'], required=['name', 'imageRef', 'flavorRef'], optional=['metadata', 'personality'])
    if 'personality' in body['server']:
        for pfile in body['server']['personality']:
            fakes.assert_has_keys(pfile, required=['path', 'contents'])
    if 'return_reservation_id' in body['server'].keys() and body['server']['return_reservation_id']:
        return {'reservation_id': 'r-3fhpjulh'}
    if body['server']['name'] == 'some-bad-server':
        body = self.server_1235
    else:
        body = self.server_1234
    return {'server': body}