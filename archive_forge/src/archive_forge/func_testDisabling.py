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
def testDisabling(self):
    """Tests enabling/disabling of metrics collection."""
    self.assertEqual(self.collector, MetricsCollector.GetCollector())
    with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': '1', 'GA_CID': '555'}):
        MetricsCollector._CheckAndSetDisabledCache()
        self.assertFalse(MetricsCollector._disabled_cache)
        self.assertEqual(self.collector, MetricsCollector.GetCollector())
    with mock.patch('boto.config.getbool', return_value=True):
        MetricsCollector._CheckAndSetDisabledCache()
        self.assertTrue(MetricsCollector._disabled_cache)
        self.assertEqual(None, MetricsCollector.GetCollector())
    with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': '1', 'GA_CID': ''}):
        MetricsCollector._CheckAndSetDisabledCache()
        self.assertTrue(MetricsCollector._disabled_cache)
        self.assertEqual(None, MetricsCollector.GetCollector())
    with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': ''}):
        with mock.patch('os.path.exists', return_value=False):
            MetricsCollector._CheckAndSetDisabledCache()
            self.assertTrue(MetricsCollector._disabled_cache)
            self.assertEqual(None, MetricsCollector.GetCollector())
    with mock.patch.dict(os.environ, values={'CLOUDSDK_WRAPPER': ''}):
        with mock.patch('os.path.exists', return_value=True):
            if six.PY2:
                builtin_open = '__builtin__.open'
            else:
                builtin_open = 'builtins.open'
            with mock.patch(builtin_open) as mock_open:
                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.read.return_value = metrics._DISABLED_TEXT
                MetricsCollector._CheckAndSetDisabledCache()
                self.assertTrue(MetricsCollector._disabled_cache)
                self.assertEqual(None, MetricsCollector.GetCollector())
                mock_open.return_value.read.return_value = 'mock_cid'
                MetricsCollector._CheckAndSetDisabledCache()
                self.assertFalse(MetricsCollector._disabled_cache)
                self.assertEqual(self.collector, MetricsCollector.GetCollector())
                self.assertEqual(2, len(mock_open.call_args_list))
                self.assertEqual(2, len(mock_open.return_value.read.call_args_list))