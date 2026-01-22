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
@_withCacheness
def test_detectFilesRemoved(self) -> None:
    """
        Check that when a dropin file is removed, L{plugin.getPlugins} doesn't
        return it anymore.
        """
    FilePath(__file__).sibling('plugin_extra1.py').copyTo(self.package.child('pluginextra.py'))
    try:
        list(plugin.getPlugins(ITestPlugin, self.module))
    finally:
        self._unimportPythonModule(sys.modules['mypackage.pluginextra'], True)
    plgs = list(plugin.getPlugins(ITestPlugin, self.module))
    self.assertEqual(1, len(plgs))