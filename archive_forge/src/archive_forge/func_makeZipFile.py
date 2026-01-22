import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def makeZipFile(self, contents, directory=''):
    """
        Makes a zip file archive containing len(contents) files.  Contents
        should be a list of strings, each string being the content of one file.
        """
    zpfilename = self.testdir.child('zipfile.zip').path
    with zipfile.ZipFile(zpfilename, 'w') as zpfile:
        for i, content in enumerate(contents):
            filename = str(i)
            if directory:
                filename = directory + '/' + filename
            zpfile.writestr(filename, content)
    return zpfilename