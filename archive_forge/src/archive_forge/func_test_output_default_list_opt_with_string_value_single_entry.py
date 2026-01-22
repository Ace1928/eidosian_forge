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
def test_output_default_list_opt_with_string_value_single_entry(self):
    self._test_output_default_list_opt_with_string_value('foo')