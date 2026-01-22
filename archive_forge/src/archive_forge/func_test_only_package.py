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
def test_only_package(self):
    self.assertEqual([('py', '/opt/b/py')], self._get_paths('/opt/b/py'))