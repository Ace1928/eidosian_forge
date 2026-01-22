import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def test_register_remove(self):
    registry = version_info_formats.format_registry
    registry.register('testbuilder', TestBuilder, 'a simple test builder')
    self.assertIs(TestBuilder, registry.get('testbuilder'))
    self.assertEqual('a simple test builder', registry.get_help('testbuilder'))
    registry.remove('testbuilder')
    self.assertRaises(KeyError, registry.get, 'testbuilder')