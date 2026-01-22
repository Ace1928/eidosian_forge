import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
def test_deprecated_env_options(self):
    val = uuid.uuid4().hex
    with mock.patch.dict('os.environ', {'OS_TEST_OTHER': val}):
        cli._register_plugin_argparse_arguments(self.p, TesterLoader())
    opts = self.p.parse_args([])
    self.assertEqual(val, opts.os_test_opt)