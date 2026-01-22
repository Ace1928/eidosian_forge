import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest
def test_unzipIterChunky(self):
    """
        L{twisted.python.zipstream.unzipIterChunky} returns an iterator which
        must be exhausted to completely unzip the input archive.
        """
    numfiles = 10
    contents = ['This is test file %d!' % i for i in range(numfiles)]
    contents = [i.encode('ascii') for i in contents]
    zpfilename = self.makeZipFile(contents)
    list(zipstream.unzipIterChunky(zpfilename, self.unzipdir.path))
    self.assertEqual(set(self.unzipdir.listdir()), set(map(str, range(numfiles))))
    for child in self.unzipdir.children():
        num = int(child.basename())
        self.assertEqual(child.getContent(), contents[num])