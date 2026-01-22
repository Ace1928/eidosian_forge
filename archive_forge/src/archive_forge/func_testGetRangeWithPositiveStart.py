import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testGetRangeWithPositiveStart(self):
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    self.__GetFile(self.__GetRequest('fifteen_byte_file'))
    self.__download.GetRange(2)
    self.assertEqual(13, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents[2:15], self.__buffer.read())