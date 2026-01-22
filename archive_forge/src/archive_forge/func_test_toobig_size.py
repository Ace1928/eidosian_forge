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
def test_toobig_size(self):
    stderr = self.RunGsUtil(['perfdiag', '-n', '1', '-s', '3pb', '-t', 'wthru', 'gs://foobar'], expected_status=1, return_stderr=True)
    self.assertIn('in-memory tests maximum file size', stderr)