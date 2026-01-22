import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_unzipIterChunkyDirectory(self):
    """
        The path to which a file is extracted by L{zipstream.unzipIterChunky}
        is determined by joining the C{directory} argument to C{unzip} with the
        path within the archive of the file being extracted.
        """
    numfiles = 10
    contents = ['This is test file %d!' % i for i in range(numfiles)]
    contents = [i.encode('ascii') for i in contents]
    zpfilename = self.makeZipFile(contents, 'foo')
    list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
    fileContents = {str(num).encode('ascii') for num in range(numfiles)}
    self.assertEqual(set(self.unzipdir.child(b'foo').listdir()), fileContents)
    for child in self.unzipdir.child(b'foo').children():
        num = int(child.basename())
        self.assertEqual(child.getContent(), contents[num])