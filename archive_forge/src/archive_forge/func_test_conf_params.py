import urllib.parse
import uuid
from oslo_config import fixture as config
import testtools
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import oidc
from keystoneclient import session
from keystoneclient.tests.unit.v3 import utils
@testtools.skip("TypeError: __init__() got an unexpected keyword argument 'project_name'")
def test_conf_params(self):
    """Ensure OpenID Connect config options work."""
    section = uuid.uuid4().hex
    identity_provider = uuid.uuid4().hex
    protocol = uuid.uuid4().hex
    username = uuid.uuid4().hex
    password = uuid.uuid4().hex
    client_id = uuid.uuid4().hex
    client_secret = uuid.uuid4().hex
    access_token_endpoint = uuid.uuid4().hex
    self.conf_fixture.config(auth_section=section, group=self.GROUP)
    conf.register_conf_options(self.conf_fixture.conf, group=self.GROUP)
    self.conf_fixture.register_opts(oidc.OidcPassword.get_options(), group=section)
    self.conf_fixture.config(auth_plugin='v3oidcpassword', identity_provider=identity_provider, protocol=protocol, username=username, password=password, client_id=client_id, client_secret=client_secret, access_token_endpoint=access_token_endpoint, group=section)
    a = conf.load_from_conf_options(self.conf_fixture.conf, self.GROUP)
    self.assertEqual(identity_provider, a.identity_provider)
    self.assertEqual(protocol, a.protocol)
    self.assertEqual(username, a.username)
    self.assertEqual(password, a.password)
    self.assertEqual(client_id, a.client_id)
    self.assertEqual(client_secret, a.client_secret)
    self.assertEqual(access_token_endpoint, a.access_token_endpoint)