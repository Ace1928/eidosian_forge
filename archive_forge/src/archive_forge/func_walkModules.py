from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
def walkModules(self, importPackages=False):
    """
        Similar to L{iterModules}, this yields every module on the path, then every
        submodule in each package or entry.
        """
    for package in self.iterModules():
        yield from package.walkModules(importPackages=False)