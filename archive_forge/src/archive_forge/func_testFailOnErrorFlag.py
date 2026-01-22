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
@Timeout
def testFailOnErrorFlag(self):
    """Tests that fail_on_error produces the correct exception on failure."""

    def _ExpectCustomException(test_func):
        try:
            test_func()
            self.fail('Setting fail_on_error should raise any exception encountered.')
        except CustomException as e:
            pass
        except Exception as e:
            self.fail('Got unexpected error: ' + str(e))

    def _RunFailureFunc():
        command_inst = self.command_class(True)
        args = [()] * 5
        self._RunApply(_FailureFunc, args, 1, 1, command_inst=command_inst, shared_attrs=['failure_count'], fail_on_error=True)
    _ExpectCustomException(_RunFailureFunc)

    def _RunFailingIteratorFirstPosition():
        args = FailingIterator(10, [0])
        results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
        self.assertEqual(0, len(results))
    _ExpectCustomException(_RunFailingIteratorFirstPosition)

    def _RunFailingIteratorPositionMiddlePosition():
        args = FailingIterator(10, [5])
        results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
        self.assertEqual(5, len(results))
    _ExpectCustomException(_RunFailingIteratorPositionMiddlePosition)

    def _RunFailingIteratorLastPosition():
        args = FailingIterator(10, [9])
        results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
        self.assertEqual(9, len(results))
    _ExpectCustomException(_RunFailingIteratorLastPosition)

    def _RunFailingIteratorMultiplePositions():
        args = FailingIterator(10, [1, 3, 5])
        results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
        self.assertEqual(1, len(results))
    _ExpectCustomException(_RunFailingIteratorMultiplePositions)