from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import logging
import os
import pyu2f
from apitools.base.py import exceptions as apitools_exceptions
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import CloudApi
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.command import CreateOrGetGsutilLogger
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.storage_url import StorageUrlFromString
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.tests.util import GSMockBucketStorageUri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils import posix_util
from gslib.utils import system_util
from gslib.utils import hashing_helper
from gslib.utils.copy_helper import _CheckCloudHashes
from gslib.utils.copy_helper import _DelegateUploadFileToObject
from gslib.utils.copy_helper import _GetPartitionInfo
from gslib.utils.copy_helper import _SelectUploadCompressionStrategy
from gslib.utils.copy_helper import _SetContentTypeFromFile
from gslib.utils.copy_helper import ExpandUrlToSingleBlr
from gslib.utils.copy_helper import FilterExistingComponents
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import PerformParallelUploadFileToObjectArgs
from gslib.utils.copy_helper import WarnIfMvEarlyDeletionChargeApplies
from six import add_move, MovedModule
from six.moves import mock
@mock.patch('gslib.cloud_api.CloudApi')
@mock.patch('gslib.utils.copy_helper.CreateWildcardIterator')
def testContainsWildcardMatchesNotObject(self, mock_CreateWildcardIterator, mock_gsutil_api):
    storage_url = StorageUrlFromString('gs://test/helloworld')
    mock_CreateWildcardIterator.return_value = iter([BucketListingPrefix(storage_url)])
    exp_url, have_existing_dst_container = ExpandUrlToSingleBlr('gs://test/hello*/', mock_gsutil_api, 'project_id', False, CreateOrGetGsutilLogger('copy_test'))
    self.assertTrue(have_existing_dst_container)
    self.assertEqual(exp_url, storage_url)