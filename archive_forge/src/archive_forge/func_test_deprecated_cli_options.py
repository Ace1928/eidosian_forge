import argparse
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import loading
from keystoneauth1.loading import cli
from keystoneauth1.tests.unit.loading import utils
def test_deprecated_cli_options(self):
    cli._register_plugin_argparse_arguments(self.p, TesterLoader())
    val = uuid.uuid4().hex
    opts = self.p.parse_args(['--os-test-other', val])
    self.assertEqual(val, opts.os_test_opt)