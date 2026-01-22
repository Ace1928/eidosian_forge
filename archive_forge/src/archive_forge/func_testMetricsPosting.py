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
@mock.patch('time.time', new=mock.MagicMock(return_value=0))
def testMetricsPosting(self):
    """Tests the metrics posting process as performed in metrics_reporter.py."""
    metrics_file = tempfile.NamedTemporaryFile()
    metrics_file_name = metrics_file.name
    metrics_file.close()

    def MetricsTempFileCleanup(file_path):
        try:
            os.unlink(file_path)
        except OSError:
            pass
    self.addCleanup(MetricsTempFileCleanup, metrics_file_name)

    def CollectMetricAndSetLogLevel(log_level, log_file_path):
        metrics.LogCommandParams(command_name='cmd1', subcommands=['action1'], sub_opts=[('optb', ''), ('opta', '')])
        metrics.LogFatalError(gslib.exception.CommandException('test'))
        self.collector.ReportMetrics(wait_for_report=True, log_level=log_level, log_file_path=log_file_path)
        self.assertEqual([], self.collector._metrics)
    metrics.LogCommandParams(global_opts=[('-y', 'value'), ('-z', ''), ('-x', '')])
    CollectMetricAndSetLogLevel(logging.DEBUG, metrics_file.name)
    with open(metrics_file.name, 'rb') as metrics_log:
        log_text = metrics_log.read()
    if six.PY2:
        expected_response = b"Metric(endpoint=u'https://example.com', method=u'POST', body='{0}&cm2=0&ea=cmd1+action1&ec={1}&el={2}&ev=0', user_agent=u'user-agent-007')".format(GLOBAL_DIMENSIONS_URL_PARAMS, metrics._GA_COMMANDS_CATEGORY, VERSION)
    else:
        expected_response = "Metric(endpoint='https://example.com', method='POST', body='{0}&cm2=0&ea=cmd1+action1&ec={1}&el={2}&ev=0', user_agent='user-agent-007')".format(GLOBAL_DIMENSIONS_URL_PARAMS, metrics._GA_COMMANDS_CATEGORY, VERSION).encode('utf_8')
    self.assertIn(expected_response, log_text)
    self.assertIn(b'RESPONSE: 200', log_text)
    CollectMetricAndSetLogLevel(logging.INFO, metrics_file.name)
    with open(metrics_file.name, 'rb') as metrics_log:
        log_text = metrics_log.read()
    self.assertEqual(log_text, b'')
    CollectMetricAndSetLogLevel(logging.WARN, metrics_file.name)
    with open(metrics_file.name, 'rb') as metrics_log:
        log_text = metrics_log.read()
    self.assertEqual(log_text, b'')