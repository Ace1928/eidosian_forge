from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
@mock.patch('os.urandom')
def test_generate_file_data_repeat(self, mock_urandom):
    """Test that random data repeats when exhausted."""

    def urandom(length):
        return b'a' * length
    mock_urandom.side_effect = urandom
    fp = six.BytesIO()
    _GenerateFileData(fp, 8, 50, 4)
    self.assertEqual(b'aaxxaaxx', fp.getvalue())
    self.assertEqual(8, fp.tell())