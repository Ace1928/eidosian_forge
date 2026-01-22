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
def test_newPluginsOnReadOnlyPath(self) -> None:
    """
        Verify that a failure to write the dropin.cache file on a read-only
        path will not affect the list of plugins returned.

        Note: this test should pass on both Linux and Windows, but may not
        provide useful coverage on Windows due to the different meaning of
        "read-only directory".
        """
    self.unlockSystem()
    self.sysplug.child('newstuff.py').setContent(pluginFileContents('one'))
    self.lockSystem()
    sys.path.remove(self.devPath.path)
    events: list[EventDict] = []
    addObserver(events.append)
    self.addCleanup(removeObserver, events.append)
    self.assertIn('one', self.getAllPlugins())
    expected = 'Unable to write to plugin cache %s: error number %d' % (self.syscache.path, errno.EPERM)
    for event in events:
        maybeText = textFromEventDict(event)
        assert maybeText is not None
        if expected in maybeText:
            break
    else:
        self.fail('Did not observe unwriteable cache warning in log events: %r' % (events,))