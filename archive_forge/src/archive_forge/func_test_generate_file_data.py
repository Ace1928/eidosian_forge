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
def test_generate_file_data(self, mock_urandom):
    """Test the right amount of random and sequential data is generated."""

    def urandom(length):
        return b'a' * length
    mock_urandom.side_effect = urandom
    fp = six.BytesIO()
    _GenerateFileData(fp, 1000, 100, 1000)
    self.assertEqual(b'a' * 1000, fp.getvalue())
    self.assertEqual(1000, fp.tell())
    fp = six.BytesIO()
    _GenerateFileData(fp, 1000, 50, 1000)
    self.assertIn(b'a' * 500, fp.getvalue())
    self.assertIn(b'x' * 500, fp.getvalue())
    self.assertEqual(1000, fp.tell())
    fp = six.BytesIO()
    _GenerateFileData(fp, 1001, 50, 1001)
    self.assertIn(b'a' * 501, fp.getvalue())
    self.assertIn(b'x' * 500, fp.getvalue())
    self.assertEqual(1001, fp.tell())