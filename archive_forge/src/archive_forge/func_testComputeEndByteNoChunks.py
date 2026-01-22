import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def testComputeEndByteNoChunks(self):
    total_size = 100
    download = transfer.Download.FromStream(six.StringIO(), chunksize=10, total_size=total_size)
    for end in (None, 1000):
        self.assertEqual(total_size - 1, download._Download__ComputeEndByte(0, end=end, use_chunks=False), msg='Failed on end={0}'.format(end))