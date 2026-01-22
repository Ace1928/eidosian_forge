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
def test_plugins_with_the_same_name_are_not_loaded(self):
    tempattribute = '0'
    self.assertFalse(tempattribute in self.activeattributes)
    self.__class__.activeattributes[tempattribute] = []
    self.assertTrue(tempattribute in self.activeattributes)
    os.mkdir('first')
    os.mkdir('second')
    template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
    with open(os.path.join('first', 'plugin.py'), 'w') as outfile:
        outfile.write(template % (tempattribute, 'first'))
        outfile.write('\n')
    with open(os.path.join('second', 'plugin.py'), 'w') as outfile:
        outfile.write(template % (tempattribute, 'second'))
        outfile.write('\n')
    try:
        self.load_with_paths(['first', 'second'])
        self.assertEqual(['first'], self.activeattributes[tempattribute])
    finally:
        del self.activeattributes[tempattribute]