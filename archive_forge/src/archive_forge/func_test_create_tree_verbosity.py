import unittest
import os
import stat
import sys
from unittest.mock import patch
from distutils import dir_util, errors
from distutils.dir_util import (mkpath, remove_tree, create_tree, copy_tree,
from distutils import log
from distutils.tests import support
from test.support import is_emscripten, is_wasi
def test_create_tree_verbosity(self):
    create_tree(self.root_target, ['one', 'two', 'three'], verbose=0)
    self.assertEqual(self._logs, [])
    remove_tree(self.root_target, verbose=0)
    wanted = ['creating %s' % self.root_target]
    create_tree(self.root_target, ['one', 'two', 'three'], verbose=1)
    self.assertEqual(self._logs, wanted)
    remove_tree(self.root_target, verbose=0)