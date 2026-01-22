import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def pathEntryTree(self, tree):
    """
        Create some files in a hierarchy, based on a dictionary describing those
        files.  The resulting hierarchy will be placed onto sys.path for the
        duration of the test.

        @param tree: A dictionary representing a directory structure.  Keys are
            strings, representing filenames, dictionary values represent
            directories, string values represent file contents.

        @return: another dictionary similar to the input, with file content
            strings replaced with L{FilePath} objects pointing at where those
            contents are now stored.
        """

    def makeSomeFiles(pathobj, dirdict):
        pathdict = {}
        for key, value in dirdict.items():
            child = pathobj.child(key)
            if isinstance(value, bytes):
                pathdict[key] = child
                child.setContent(value)
            elif isinstance(value, dict):
                child.createDirectory()
                pathdict[key] = makeSomeFiles(child, value)
            else:
                raise ValueError('only strings and dicts allowed as values')
        return pathdict
    base = FilePath(self.mktemp().encode('utf-8'))
    base.makedirs()
    result = makeSomeFiles(base, tree)
    self.replaceSysPath([base.path.decode('utf-8')] + sys.path)
    self.replaceSysModules(sys.modules.copy())
    return result