import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testFilenameWithSpaces(self):
    self.__ResetDownload(auto_transfer=True)
    self.__GetFile(self.__GetRequest('filename with spaces'))
    file_contents = self.__GetTestdataFileContents('filename_with_spaces')
    self.assertEqual(15, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents, self.__buffer.read())