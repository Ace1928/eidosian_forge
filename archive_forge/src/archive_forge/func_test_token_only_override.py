import argparse
from unittest import mock
import uuid
from keystoneclient.auth.identity.generic import cli
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
def test_token_only_override(self):
    self.assertRaises(exceptions.CommandError, self.new_plugin, ['--os-token', uuid.uuid4().hex])