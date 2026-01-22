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
@RequiresIsolation
def testMultithreadingDoesNotLogMacOSWarning(self):
    logger = CreateOrGetGsutilLogger('FakeCommand')
    mock_log_handler = MockLoggingHandler()
    logger.addHandler(mock_log_handler)
    self._TestRecursiveDepthThreeDifferentFunctions(1, 3)
    macos_message = 'If you experience problems with multiprocessing on MacOS'
    contains_message = [message.startswith(macos_message) for message in mock_log_handler.messages['info']]
    self.assertFalse(any(contains_message))
    logger.removeHandler(mock_log_handler)