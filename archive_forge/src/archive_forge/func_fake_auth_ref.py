import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
def fake_auth_ref(fake_token, fake_service=None):
    """Create an auth_ref using keystoneauth's fixtures"""
    token_copy = copy.deepcopy(fake_token)
    token_copy['token_id'] = token_copy.pop('id')
    token = fixture.V2Token(**token_copy)
    auth_ref = access.create(body=token)
    if fake_service:
        service = token.add_service(fake_service.type, fake_service.name)
        service['id'] = fake_service.id
        for e in fake_service.endpoints:
            internal = e.get('internalURL', None)
            admin = e.get('adminURL', None)
            region = e.get('region_id') or e.get('region', '<none>')
            endpoint = service.add_endpoint(public=e['publicURL'], internal=internal, admin=admin, region=region)
            if not internal:
                endpoint['internalURL'] = None
            if not admin:
                endpoint['adminURL'] = None
    return auth_ref