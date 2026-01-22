import json
import os
import random
import string
import unittest
import six
from apitools.base.py import transfer
import storage
def testBreakAndResumeUpload(self):
    filename = 'ten_meg_file_' + ''.join(random.sample(string.ascii_letters, 5))
    size = 10 << 20
    self.__ResetUpload(size, auto_transfer=False)
    self.__upload.strategy = 'resumable'
    self.__upload.total_size = size
    request = self.__InsertRequest(filename)
    initial_response = self.__client.objects.Insert(request, upload=self.__upload)
    self.assertIsNotNone(initial_response)
    self.assertEqual(0, self.__buffer.tell())
    upload_data = json.dumps(self.__upload.serialization_data)
    second_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
    second_upload_attempt._Upload__SendChunk(0)
    self.assertEqual(second_upload_attempt.chunksize, self.__buffer.tell())
    final_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
    final_upload_attempt.StreamInChunks()
    self.assertEqual(size, self.__buffer.tell())
    object_info = self.__client.objects.Get(self.__GetRequest(filename))
    self.assertEqual(size, object_info.size)
    completed_upload_attempt = transfer.Upload.FromData(self.__buffer, upload_data, self.__upload.http)
    self.assertTrue(completed_upload_attempt.complete)
    completed_upload_attempt.StreamInChunks()
    object_info = self.__client.objects.Get(self.__GetRequest(filename))
    self.assertEqual(size, object_info.size)