import base64
import glob
import os
import pickle
from twisted.python.filepath import FilePath
def copyTo(self, path):
    """
        Copy the contents of this dirdbm to the dirdbm at C{path}.

        @type path: L{str}
        @param path: The path of the dirdbm to copy to.  If a dirdbm
        exists at the destination path, it is cleared first.

        @rtype: C{DirDBM}
        @return: The dirdbm this dirdbm was copied to.
        """
    path = FilePath(path)
    assert path != self._dnamePath
    d = self.__class__(path.path)
    d.clear()
    for k in self.keys():
        d[k] = self[k]
    return d