from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
def test_mapped_with_remote_user(self):
    method_name = 'saml2'
    auth_data = {'methods': [method_name]}
    auth_data[method_name] = {'protocol': method_name}
    auth_data = {'identity': auth_data}
    auth_context = auth.core.AuthContext(method_names=[], user_id=uuid.uuid4().hex)
    self.useFixture(auth_plugins.LoadAuthPlugins(method_name))
    with mock.patch.object(auth.plugins.mapped.Mapped, 'authenticate', return_value=None) as authenticate:
        auth_info = auth.core.AuthInfo.create(auth_data)
        with self.make_request(environ={'REMOTE_USER': 'foo@idp.com'}):
            authentication.authenticate(auth_info, auth_context)
        (auth_payload,), kwargs = authenticate.call_args
        self.assertEqual(method_name, auth_payload['protocol'])