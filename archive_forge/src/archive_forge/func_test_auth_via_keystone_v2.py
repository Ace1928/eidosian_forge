import os
from urllib import parse
import tempest.lib.cli.base
from novaclient import client
from novaclient.tests.functional import base
def test_auth_via_keystone_v2(self):
    session = self.keystone.session
    version = (2, 0)
    if not base.is_keystone_version_available(session, version):
        self.skipTest('Identity API version 2.0 is not available.')
    self.nova_auth_with_password('list', identity_api_version='2.0')
    self.nova_auth_with_token(identity_api_version='2.0')