import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
def getModules(self):
    """
        Return matching module names for files listed in C{self.files}.
        """
    return [self._toModuleName(filename) for filename, code in self.files]