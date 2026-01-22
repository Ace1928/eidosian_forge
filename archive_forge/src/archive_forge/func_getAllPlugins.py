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
def getAllPlugins(self) -> list[str]:
    """
        Get all the plugins loadable from our dummy package, and return their
        short names.
        """
    import plugindummy.plugins
    x = list(plugin.getPlugins(ITestPlugin, plugindummy.plugins))
    return [plug.__name__ for plug in x]