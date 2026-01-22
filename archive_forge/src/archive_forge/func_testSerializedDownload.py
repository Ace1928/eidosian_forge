import json
import os
import unittest
import six
from apitools.base.py import exceptions
import storage
def testSerializedDownload(self):

    def _ProgressCallback(unused_response, download_object):
        print('Progress %s' % download_object.progress)
    file_contents = self.__GetTestdataFileContents('fifteen_byte_file')
    object_name = os.path.join(self._TESTDATA_PREFIX, 'fifteen_byte_file')
    request = storage.StorageObjectsGetRequest(bucket=self._DEFAULT_BUCKET, object=object_name)
    response = self.__client.objects.Get(request)
    self.__buffer = six.StringIO()
    download_data = json.dumps({'auto_transfer': False, 'progress': 0, 'total_size': response.size, 'url': response.mediaLink})
    self.__download = storage.Download.FromData(self.__buffer, download_data, http=self.__client.http)
    self.__download.StreamInChunks(callback=_ProgressCallback)
    self.assertEqual(15, self.__buffer.tell())
    self.__buffer.seek(0)
    self.assertEqual(file_contents, self.__buffer.read(15))