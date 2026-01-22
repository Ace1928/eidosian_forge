import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_get_consumed_file_size(self):
    size, consumed = (98304, 304)
    file_obj = io.StringIO('X' * size)
    file_obj.seek(consumed)
    try:
        self.assertEqual(size, utils.get_file_size(file_obj))
        self.assertEqual(consumed, file_obj.tell())
    finally:
        file_obj.close()