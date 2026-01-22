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
def test_many_at(self):
    self.assertEqual([('church', os.path.abspath('StMichael@Plea@Norwich'))], self._get_paths('church@StMichael@Plea@Norwich'))