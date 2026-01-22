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
def test_generator_call_with_no_arguments_raises_system_exit(self):
    testargs = ['oslo-config-generator']
    with mock.patch('sys.argv', testargs):
        self.assertRaises(SystemExit, generator.main, [])