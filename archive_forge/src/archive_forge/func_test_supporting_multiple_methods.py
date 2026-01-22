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
def test_supporting_multiple_methods(self):
    method_names = ('saml2', 'openid', 'x509', 'mapped')
    self.useFixture(auth_plugins.LoadAuthPlugins(*method_names))
    for method_name in method_names:
        self._test_mapped_invocation_with_method_name(method_name)