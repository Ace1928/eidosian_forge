from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import pprint
import six
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from boto.storage_uri import BucketStorageUri
from gslib import metrics
from gslib import VERSION
from gslib.cs_api_map import ApiSelector
import gslib.exception
from gslib.gcs_json_api import GcsJsonApi
from gslib.metrics import MetricsCollector
from gslib.metrics_tuple import Metric
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SkipForParFile
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from six import add_move, MovedModule
from six.moves import mock
def testExceptionCatchingDecorator(self):
    """Tests the exception catching decorator CaptureAndLogException."""
    mock_exc_fn = mock.MagicMock(__name__=str('mock_exc_fn'), side_effect=Exception())
    wrapped_fn = metrics.CaptureAndLogException(mock_exc_fn)
    wrapped_fn()
    debug_messages = self.log_handler.messages['debug']
    self.assertIn('Exception captured in mock_exc_fn during metrics collection', debug_messages[0])
    self.log_handler.reset()
    self.assertEqual(1, mock_exc_fn.call_count)
    mock_err_fn = mock.MagicMock(__name__=str('mock_err_fn'), side_effect=TypeError())
    wrapped_fn = metrics.CaptureAndLogException(mock_err_fn)
    wrapped_fn()
    self.assertEqual(1, mock_err_fn.call_count)
    debug_messages = self.log_handler.messages['debug']
    self.assertIn('Exception captured in mock_err_fn during metrics collection', debug_messages[0])
    self.log_handler.reset()
    with mock.patch.object(MetricsCollector, 'GetCollector', return_value='not a collector'):
        metrics.Shutdown()
        metrics.LogCommandParams()
        metrics.LogRetryableError()
        metrics.LogFatalError()
        metrics.LogPerformanceSummaryParams()
        metrics.CheckAndMaybePromptForAnalyticsEnabling('invalid argument')
        debug_messages = self.log_handler.messages['debug']
        message_index = 0
        for func_name in ('Shutdown', 'LogCommandParams', 'LogRetryableError', 'LogFatalError', 'LogPerformanceSummaryParams', 'CheckAndMaybePromptForAnalyticsEnabling'):
            self.assertIn('Exception captured in %s during metrics collection' % func_name, debug_messages[message_index])
            message_index += 1
        self.log_handler.reset()