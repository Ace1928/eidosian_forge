import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def testStreamMedia(self):
    filename = 'ten_meg_file'
    size = 10 << 20
    self.__ResetUpload(size, auto_transfer=False)
    self.__upload.strategy = 'resumable'
    self.__upload.total_size = size
    request = self.__InsertRequest(filename)
    initial_response = self.__client.objects.Insert(request, upload=self.__upload)
    self.assertIsNotNone(initial_response)
    self.assertEqual(0, self.__buffer.tell())
    self.__upload.StreamMedia()
    self.assertEqual(size, self.__buffer.tell())