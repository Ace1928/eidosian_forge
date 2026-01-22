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
@mock.patch('time.time', new=mock.MagicMock(return_value=posix_util.ConvertDatetimeToPOSIX(_PI_DAY)))
def testWarnIfMvEarlyDeletionChargeApplies(self):
    """Tests that WarnIfEarlyDeletionChargeApplies warns when appropriate."""
    test_logger = logging.Logger('test')
    src_url = StorageUrlFromString('gs://bucket/object')
    for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=29, hours=23)):
        recent_nearline_obj = apitools_messages.Object(storageClass='NEARLINE', timeCreated=object_time_created)
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            WarnIfMvEarlyDeletionChargeApplies(src_url, recent_nearline_obj, test_logger)
            mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'nearline', src_url.url_string, 30)
    for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=89, hours=23)):
        recent_nearline_obj = apitools_messages.Object(storageClass='COLDLINE', timeCreated=object_time_created)
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            WarnIfMvEarlyDeletionChargeApplies(src_url, recent_nearline_obj, test_logger)
            mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'coldline', src_url.url_string, 90)
    for object_time_created in (self._PI_DAY, self._PI_DAY - datetime.timedelta(days=364, hours=23)):
        recent_archive_obj = apitools_messages.Object(storageClass='ARCHIVE', timeCreated=object_time_created)
        with mock.patch.object(test_logger, 'warn') as mocked_warn:
            WarnIfMvEarlyDeletionChargeApplies(src_url, recent_archive_obj, test_logger)
            mocked_warn.assert_called_with('Warning: moving %s object %s may incur an early deletion charge, because the original object is less than %s days old according to the local system time.', 'archive', src_url.url_string, 365)
    with mock.patch.object(test_logger, 'warn') as mocked_warn:
        old_nearline_obj = apitools_messages.Object(storageClass='NEARLINE', timeCreated=self._PI_DAY - datetime.timedelta(days=30, seconds=1))
        WarnIfMvEarlyDeletionChargeApplies(src_url, old_nearline_obj, test_logger)
        mocked_warn.assert_not_called()
    with mock.patch.object(test_logger, 'warn') as mocked_warn:
        old_coldline_obj = apitools_messages.Object(storageClass='COLDLINE', timeCreated=self._PI_DAY - datetime.timedelta(days=90, seconds=1))
        WarnIfMvEarlyDeletionChargeApplies(src_url, old_coldline_obj, test_logger)
        mocked_warn.assert_not_called()
    with mock.patch.object(test_logger, 'warn') as mocked_warn:
        old_archive_obj = apitools_messages.Object(storageClass='ARCHIVE', timeCreated=self._PI_DAY - datetime.timedelta(days=365, seconds=1))
        WarnIfMvEarlyDeletionChargeApplies(src_url, old_archive_obj, test_logger)
        mocked_warn.assert_not_called()
    with mock.patch.object(test_logger, 'warn') as mocked_warn:
        not_old_enough_nearline_obj = apitools_messages.Object(storageClass='STANDARD', timeCreated=self._PI_DAY)
        WarnIfMvEarlyDeletionChargeApplies(src_url, not_old_enough_nearline_obj, test_logger)
        mocked_warn.assert_not_called()