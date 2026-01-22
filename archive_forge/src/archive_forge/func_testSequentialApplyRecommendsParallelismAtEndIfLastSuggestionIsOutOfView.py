from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import functools
import os
import signal
import six
import threading
import textwrap
import time
from unittest import mock
import boto
from boto.storage_uri import BucketStorageUri
from boto.storage_uri import StorageUri
from gslib import cs_api_map
from gslib import command
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import DummyArgChecker
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import RequiresIsolation
from gslib.tests.util import unittest
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
@mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_THRESHOLD', 100)
@mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_FREQUENCY', 10)
@mock.patch.object(command, 'GetTermLines', return_value=2)
def testSequentialApplyRecommendsParallelismAtEndIfLastSuggestionIsOutOfView(self, mock_get_term_lines):
    logger = CreateOrGetGsutilLogger('FakeCommand')
    mock_log_handler = MockLoggingHandler()
    logger.addHandler(mock_log_handler)
    self._RunApply(_ReturnOneValue, range(22), process_count=1, thread_count=1)
    contains_message = [message == PARALLEL_PROCESSING_MESSAGE for message in mock_log_handler.messages['info']]
    self.assertEqual(sum(contains_message), 3)
    logger.removeHandler(mock_log_handler)