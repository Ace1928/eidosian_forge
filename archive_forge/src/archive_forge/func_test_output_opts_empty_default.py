import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
def test_output_opts_empty_default(self):
    config = [('namespace1', [('alpha', [])])]
    groups = generator._get_groups(config)
    fd, tmp_file = tempfile.mkstemp()
    with open(tmp_file, 'w+') as f:
        formatter = build_formatter(f)
        generator._output_opts(formatter, 'DEFAULT', groups.pop('DEFAULT'))
    expected = '[DEFAULT]\n'
    with open(tmp_file, 'r') as f:
        actual = f.read()
    self.assertEqual(expected, actual)