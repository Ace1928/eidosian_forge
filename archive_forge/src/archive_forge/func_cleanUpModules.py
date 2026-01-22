import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
def cleanUpModules(self):
    modules = self.getModules()
    modules.sort()
    modules.reverse()
    for module in modules:
        try:
            del sys.modules[module]
        except KeyError:
            pass