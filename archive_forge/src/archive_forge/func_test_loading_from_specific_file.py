import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
def test_loading_from_specific_file(self):
    plugin_dir = 'non-standard-dir'
    plugin_file_name = 'iamtestfoo.py'
    plugin_path = osutils.pathjoin(plugin_dir, plugin_file_name)
    source = '"""This is the doc for {}"""\ndir_source = \'{}\'\n'.format('test_foo', plugin_path)
    self.create_plugin('test_foo', source=source, dir=plugin_dir, file_name=plugin_file_name)
    self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@%s' % plugin_path)
    self.load_with_paths(['standard'])
    self.assertTestFooLoadedFrom(plugin_path)