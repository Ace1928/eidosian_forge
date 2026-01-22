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
def test_hiddenPackageSamePluginModuleNameObscured(self) -> None:
    """
        Only plugins from the first package in sys.path should be returned by
        getPlugins in the case where there are two Python packages by the same
        name installed, each with a plugin module by a single name.
        """
    root = FilePath(self.mktemp())
    root.makedirs()
    firstDirectory = self.createDummyPackage(root, 'first', 'someplugin')
    secondDirectory = self.createDummyPackage(root, 'second', 'someplugin')
    sys.path.append(firstDirectory.path)
    sys.path.append(secondDirectory.path)
    import dummy.plugins
    plugins = list(plugin.getPlugins(ITestPlugin, dummy.plugins))
    self.assertEqual(['first'], [p.__name__ for p in plugins])