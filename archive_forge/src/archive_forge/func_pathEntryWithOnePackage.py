import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath
def pathEntryWithOnePackage(self, pkgname: str='test_package') -> FilePath[str]:
    """
        Generate a L{FilePath} with one package, named C{pkgname}, on it, and
        return the L{FilePath} of the path entry.
        """
    entry = FilePath(self.mktemp())
    pkg = entry.child('test_package')
    pkg.makedirs()
    pkg.child('__init__.py').setContent(b'')
    return entry