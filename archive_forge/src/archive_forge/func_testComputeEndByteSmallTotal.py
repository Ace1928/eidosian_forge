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
def testComputeEndByteSmallTotal(self):
    total_size = 100
    download = transfer.Download.FromStream(six.StringIO(), total_size=total_size)
    for start in (0, 10):
        self.assertEqual(total_size - 1, download._Download__ComputeEndByte(start), msg='Failed on start={0}'.format(start))