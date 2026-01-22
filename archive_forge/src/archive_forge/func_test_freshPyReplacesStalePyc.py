from __future__ import annotations
import compileall
import errno
import functools
import os
import sys
import time
from importlib import invalidate_caches as invalidateImportCaches
from types import ModuleType
from typing import Callable, TypedDict, TypeVar
from zope.interface import Interface
from twisted import plugin
from twisted.python.filepath import FilePath
from twisted.python.log import EventDict, addObserver, removeObserver, textFromEventDict
from twisted.trial import unittest
from twisted.plugin import pluginPackagePaths
def test_freshPyReplacesStalePyc(self) -> None:
    """
        Verify that if a stale .pyc file on the PYTHONPATH is replaced by a
        fresh .py file, the plugins in the new .py are picked up rather than
        the stale .pyc, even if the .pyc is still around.
        """
    mypath = self.appPackage.child('stale.py')
    mypath.setContent(pluginFileContents('one'))
    x = time.time() - 1000
    os.utime(mypath.path, (x, x))
    pyc = mypath.sibling('stale.pyc')
    extra = _HasBoolLegacyKey(legacy=True)
    compileall.compile_dir(self.appPackage.path, quiet=1, **extra)
    os.utime(pyc.path, (x, x))
    mypath.remove()
    self.resetEnvironment()
    self.assertIn('one', self.getAllPlugins())
    self.failIfIn('two', self.getAllPlugins())
    self.resetEnvironment()
    mypath.setContent(pluginFileContents('two'))
    self.failIfIn('one', self.getAllPlugins())
    self.assertIn('two', self.getAllPlugins())