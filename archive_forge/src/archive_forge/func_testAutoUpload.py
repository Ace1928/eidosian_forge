import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def testAutoUpload(self):
    filename = 'ten_meg_file'
    size = 10 << 20
    self.__ResetUpload(size)
    request = self.__InsertRequest(filename)
    response = self.__InsertFile(filename, request=request)
    self.assertEqual(size, response.size)