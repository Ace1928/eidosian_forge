import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
def loadTestsFromModuleName(self, name):
    if self.needs_module(name):
        return TestLoader.loadTestsFromModuleName(self, name)
    else:
        return self.suiteClass()