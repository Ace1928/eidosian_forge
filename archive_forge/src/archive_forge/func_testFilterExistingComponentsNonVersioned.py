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
def testFilterExistingComponentsNonVersioned(self):
    """Tests upload with a variety of component states."""
    mock_api = MockCloudApi()
    bucket_name = self.MakeTempName('bucket')
    tracker_file = self.CreateTempFile(file_name='foo', contents=b'asdf')
    tracker_file_lock = parallelism_framework_util.CreateLock()
    content_type = 'ContentType'
    storage_class = 'StorageClass'
    fpath_uploaded_correctly = self.CreateTempFile(file_name='foo1', contents=b'1')
    fpath_uploaded_correctly_url = StorageUrlFromString(str(fpath_uploaded_correctly))
    object_uploaded_correctly_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_uploaded_correctly))
    with open(fpath_uploaded_correctly, 'rb') as f_in:
        fpath_uploaded_correctly_md5 = _CalculateB64EncodedMd5FromContents(f_in)
    mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_uploaded_correctly, md5Hash=fpath_uploaded_correctly_md5), contents=b'1')
    args_uploaded_correctly = PerformParallelUploadFileToObjectArgs(fpath_uploaded_correctly, 0, 1, fpath_uploaded_correctly_url, object_uploaded_correctly_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_not_uploaded = self.CreateTempFile(file_name='foo2', contents=b'2')
    fpath_not_uploaded_url = StorageUrlFromString(str(fpath_not_uploaded))
    object_not_uploaded_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_not_uploaded))
    args_not_uploaded = PerformParallelUploadFileToObjectArgs(fpath_not_uploaded, 0, 1, fpath_not_uploaded_url, object_not_uploaded_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_wrong_contents = self.CreateTempFile(file_name='foo4', contents=b'4')
    fpath_wrong_contents_url = StorageUrlFromString(str(fpath_wrong_contents))
    object_wrong_contents_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_wrong_contents))
    with open(self.CreateTempFile(contents=b'_'), 'rb') as f_in:
        fpath_wrong_contents_md5 = _CalculateB64EncodedMd5FromContents(f_in)
    mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name=fpath_wrong_contents, md5Hash=fpath_wrong_contents_md5), contents=b'1')
    args_wrong_contents = PerformParallelUploadFileToObjectArgs(fpath_wrong_contents, 0, 1, fpath_wrong_contents_url, object_wrong_contents_url, '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_remote_deleted = self.CreateTempFile(file_name='foo5', contents=b'5')
    fpath_remote_deleted_url = StorageUrlFromString(str(fpath_remote_deleted))
    args_remote_deleted = PerformParallelUploadFileToObjectArgs(fpath_remote_deleted, 0, 1, fpath_remote_deleted_url, '', '', content_type, storage_class, tracker_file, tracker_file_lock, None, False)
    fpath_no_longer_used = self.CreateTempFile(file_name='foo6', contents=b'6')
    with open(fpath_no_longer_used, 'rb') as f_in:
        file_md5 = _CalculateB64EncodedMd5FromContents(f_in)
    mock_api.MockCreateObjectWithMetadata(apitools_messages.Object(bucket=bucket_name, name='foo6', md5Hash=file_md5), contents=b'6')
    dst_args = {fpath_uploaded_correctly: args_uploaded_correctly, fpath_not_uploaded: args_not_uploaded, fpath_wrong_contents: args_wrong_contents, fpath_remote_deleted: args_remote_deleted}
    existing_components = [ObjectFromTracker(fpath_uploaded_correctly, ''), ObjectFromTracker(fpath_wrong_contents, ''), ObjectFromTracker(fpath_remote_deleted, ''), ObjectFromTracker(fpath_no_longer_used, '')]
    bucket_url = StorageUrlFromString('%s://%s' % (self.default_provider, bucket_name))
    components_to_upload, uploaded_components, existing_objects_to_delete = FilterExistingComponents(dst_args, existing_components, bucket_url, mock_api)
    uploaded_components = [i[0] for i in uploaded_components]
    for arg in [args_not_uploaded, args_wrong_contents, args_remote_deleted]:
        self.assertTrue(arg in components_to_upload)
    self.assertEqual(1, len(uploaded_components))
    self.assertEqual(args_uploaded_correctly.dst_url.url_string, uploaded_components[0].url_string)
    self.assertEqual(1, len(existing_objects_to_delete))
    no_longer_used_url = StorageUrlFromString('%s://%s/%s' % (self.default_provider, bucket_name, fpath_no_longer_used))
    self.assertEqual(no_longer_used_url.url_string, existing_objects_to_delete[0].url_string)