import argparse
from unittest import mock
import uuid
from keystoneclient.auth.identity.generic import cli
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
def test_endpoint_override(self):
    password = uuid.uuid4().hex
    url = uuid.uuid4().hex
    p = self.new_plugin(['--os-auth-url', 'url', '--os-endpoint', url, '--os-password', password])
    self.assertEqual(url, p.get_endpoint(None))
    self.assertEqual(password, p._password)