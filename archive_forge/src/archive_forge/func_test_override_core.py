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
def test_override_core(self):
    self.check_path(['mycore', self.user, self.site], ['mycore', '-core', '+user', '+site'])
    self.check_path(['mycore', self.site], ['mycore', '-core'])