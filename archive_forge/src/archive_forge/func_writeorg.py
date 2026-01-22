import os
import sys
import py
import tempfile
def writeorg(self, data):
    """ write a string to the original file descriptor
        """
    tempfp = tempfile.TemporaryFile()
    try:
        os.dup2(self._savefd, tempfp.fileno())
        tempfp.write(data)
    finally:
        tempfp.close()