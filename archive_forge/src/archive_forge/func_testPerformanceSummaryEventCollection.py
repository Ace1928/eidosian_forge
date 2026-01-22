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
def testPerformanceSummaryEventCollection(self):
    """Test the collection of PerformanceSummary GA events."""
    self.collector.ga_params[metrics._GA_LABEL_MAP['Command Name']] = 'cp'
    with mock.patch('gslib.metrics.system_util.GetDiskCounters', return_value={'fake-disk': (0, 0, 0, 0, 0, 0)}):
        metrics.LogPerformanceSummaryParams(uses_fan=True, uses_slice=True, avg_throughput=10, is_daisy_chain=True, has_file_dst=False, has_cloud_dst=True, has_file_src=False, has_cloud_src=True, total_bytes_transferred=100, total_elapsed_time=10, thread_idle_time=40, thread_execution_time=10, num_processes=2, num_threads=3, num_objects_transferred=3, provider_types=['gs'])
    service_retry_msg = RetryableErrorMessage(apitools_exceptions.CommunicationError(), 0)
    network_retry_msg = RetryableErrorMessage(socket.error(), 0)
    metrics.LogRetryableError(service_retry_msg)
    metrics.LogRetryableError(network_retry_msg)
    metrics.LogRetryableError(network_retry_msg)
    start_file_msg = FileMessage('src', 'dst', 0, size=100)
    end_file_msg = FileMessage('src', 'dst', 10, finished=True)
    start_file_msg.thread_id = end_file_msg.thread_id = 1
    start_file_msg.process_id = end_file_msg.process_id = 1
    metrics.LogPerformanceSummaryParams(file_message=start_file_msg)
    metrics.LogPerformanceSummaryParams(file_message=end_file_msg)
    self.assertEqual(self.collector.perf_sum_params.thread_throughputs[1, 1].GetThroughput(), 10)
    with mock.patch('gslib.metrics.system_util.GetDiskCounters', return_value={'fake-disk': (0, 0, 0, 0, 10, 10)}):
        self.collector._CollectPerformanceSummaryMetric()
    metric_body = self.collector._metrics[0].body
    label_and_value_pairs = [('Event Category', metrics._GA_PERFSUM_CATEGORY), ('Event Action', 'CloudToCloud%2CDaisyChain'), ('Execution Time', '10'), ('Parallelism Strategy', 'both'), ('Source URL Type', 'cloud'), ('Provider Types', 'gs'), ('Num Processes', '2'), ('Num Threads', '3'), ('Number of Files/Objects Transferred', '3'), ('Size of Files/Objects Transferred', '100'), ('Average Overall Throughput', '10'), ('Num Retryable Service Errors', '1'), ('Num Retryable Network Errors', '2'), ('Thread Idle Time Percent', '0.8'), ('Slowest Thread Throughput', '10'), ('Fastest Thread Throughput', '10')]
    if IS_LINUX:
        label_and_value_pairs.append(('Disk I/O Time', '20'))
    for label, exp_value in label_and_value_pairs:
        self.assertIn('{0}={1}'.format(metrics._GA_LABEL_MAP[label], exp_value), metric_body)