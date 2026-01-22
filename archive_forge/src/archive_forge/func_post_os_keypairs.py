from novaclient import api_versions
from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
def post_os_keypairs(request, context):
    body = request.json()
    assert list(body) == ['keypair']
    if api_version >= api_versions.APIVersion('2.92'):
        required = ['name', 'public_key']
    else:
        required = ['name']
    fakes.assert_has_keys(body['keypair'], required=required)
    return {'keypair': keypair}