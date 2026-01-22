import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testSmallChunksizes(self):
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    request = self.__GetRequest('fifteen_byte_file')
    for chunksize in (2, 3, 15, 100):
        self.__ResetDownload()
        self.__download.chunksize = chunksize
        self.__GetAndStream(request)
        self.assertEqual(15, self.__buffer.tell())
        self.__buffer.seek(0)
        self.assertEqual(file_contents, self.__buffer.read(15))