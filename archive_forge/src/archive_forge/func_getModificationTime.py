import base64
import glob
import os
import pickle
from twisted.python.filepath import FilePath
def getModificationTime(self, key):
    """
        Returns modification time of an entry.

        @return: Last modification date (seconds since epoch) of entry C{key}
        @raise KeyError: Raised when there is no such key
        """
    if not type(key) == bytes:
        raise TypeError('DirDBM key must be bytes')
    path = self._dnamePath.child(self._encode(key))
    if path.isfile():
        return path.getModificationTime()
    else:
        raise KeyError(key)