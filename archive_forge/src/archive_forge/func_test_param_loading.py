import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_param_loading(self, m):
    name = uuid.uuid4().hex
    argv = ['--os-auth-type', name, '--os-a-int', str(self.a_int), '--os-a-float', str(self.a_float), '--os-a-bool', str(self.a_bool)]
    klass = loading.register_auth_argparse_arguments(self.p, argv)
    self.assertIsInstance(klass, utils.MockLoader)
    opts = self.p.parse_args(argv)
    self.assertEqual(name, opts.os_auth_type)
    a = loading.load_auth_from_argparse_arguments(opts)
    self.assertTestVals(a)
    self.assertEqual(name, opts.os_auth_type)
    self.assertEqual(str(self.a_int), opts.os_a_int)
    self.assertEqual(str(self.a_float), opts.os_a_float)
    self.assertEqual(str(self.a_bool), opts.os_a_bool)