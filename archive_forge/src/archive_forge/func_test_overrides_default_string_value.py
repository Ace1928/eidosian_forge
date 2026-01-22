import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_overrides_default_string_value(self, m):
    name = uuid.uuid4().hex
    default = uuid.uuid4().hex
    argv = ['--os-auth-type', name]
    klass = loading.register_auth_argparse_arguments(self.p, argv, default=default)
    self.assertIsInstance(klass, utils.MockLoader)
    m.assert_called_once_with(name)