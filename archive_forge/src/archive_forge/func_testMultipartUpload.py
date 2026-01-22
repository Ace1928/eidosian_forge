import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def testMultipartUpload(self):
    filename = 'fifteen_byte_file'
    self.__ResetUpload(15)
    request = self.__InsertRequest(filename)
    request.object = storage.Object(contentLanguage='en')
    response = self.__InsertFile(filename, request=request)
    self.assertEqual(15, response.size)
    self.assertEqual('en', response.contentLanguage)