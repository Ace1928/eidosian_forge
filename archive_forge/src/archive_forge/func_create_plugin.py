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
def create_plugin(self, name, source=None, dir='.', file_name=None):
    if source is None:
        source = '"""This is the doc for %s"""\n' % name
    if file_name is None:
        file_name = name + '.py'
    path = osutils.pathjoin(dir, file_name)
    with open(path, 'w') as f:
        f.write(source + '\n')