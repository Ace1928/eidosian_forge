import argparse
from unittest import mock
import uuid
from keystoneclient.auth.identity.generic import cli
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
def test_token_endpoint_override(self):
    token = uuid.uuid4().hex
    endpoint = uuid.uuid4().hex
    p = self.new_plugin(['--os-endpoint', endpoint, '--os-token', token])
    self.assertEqual(endpoint, p.get_endpoint(None))
    self.assertEqual(token, p.get_token(None))