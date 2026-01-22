from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ast
import base64
import binascii
import datetime
import gzip
import logging
import os
import pickle
import pkgutil
import random
import re
import stat
import string
import sys
import threading
from unittest import mock
from apitools.base.py import exceptions as apitools_exceptions
import boto
from boto import storage_uri
from boto.exception import ResumableTransferDisposition
from boto.exception import StorageResponseError
from boto.storage_uri import BucketStorageUri
from gslib import command
from gslib import exception
from gslib import name_expansion
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import _DEFAULT_DOWNLOAD_CHUNK_SIZE
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import InvalidUrlError
from gslib.gcs_json_api import GcsJsonApi
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
from gslib.tests.rewrite_helper import EnsureRewriteResumeCallbackHandler
from gslib.tests.rewrite_helper import HaltingRewriteCallbackHandler
from gslib.tests.rewrite_helper import RewriteHaltException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import NotParallelizable
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import HaltingCopyCallbackHandler
from gslib.tests.util import HaltOneComponentCopyCallbackHandler
from gslib.tests.util import HAS_GS_PORT
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import KmsTestingResources
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import TEST_ENCRYPTION_KEY1_SHA256_B64
from gslib.tests.util import TEST_ENCRYPTION_KEY2
from gslib.tests.util import TEST_ENCRYPTION_KEY3
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import GetSlicedDownloadTrackerFilePaths
from gslib.ui_controller import BytesToFixedWidthString
from gslib.utils import hashing_helper
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.constants import UTF8
from gslib.utils.copy_helper import GetTrackerFilePath
from gslib.utils.copy_helper import PARALLEL_UPLOAD_STATIC_SALT
from gslib.utils.copy_helper import PARALLEL_UPLOAD_TEMP_NAMESPACE
from gslib.utils.copy_helper import TrackerFileType
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import ValidatePOSIXMode
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.text_util import get_random_ascii_chars
from gslib.utils.unit_util import EIGHT_MIB
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from gslib.utils import shim_util
import six
from six.moves import http_client
from six.moves import range
from six.moves import xrange
@SkipForS3('No resumable upload support for S3.')
def test_cp_composite_encrypted_upload_resume(self):
    """Tests that an encrypted composite upload resumes successfully."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('gsutil does not support encryption with the XML API')
    bucket_uri = self.CreateBucket()
    dst_url = StorageUrlFromString(suri(bucket_uri, 'foo'))
    file_contents = b'foobar'
    file_name = 'foobar'
    source_file = self.CreateTempFile(contents=file_contents, file_name=file_name)
    src_url = StorageUrlFromString(source_file)
    tracker_file_name = GetTrackerFilePath(dst_url, TrackerFileType.PARALLEL_UPLOAD, self.test_api, src_url)
    tracker_prefix = '123'
    encoded_name = (PARALLEL_UPLOAD_STATIC_SALT + source_file).encode(UTF8)
    content_md5 = GetMd5()
    content_md5.update(encoded_name)
    digest = content_md5.hexdigest()
    component_object_name = tracker_prefix + PARALLEL_UPLOAD_TEMP_NAMESPACE + digest + '_0'
    component_size = 3
    object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name=component_object_name, contents=file_contents[:component_size], encryption_key=TEST_ENCRYPTION_KEY1)
    existing_component = ObjectFromTracker(component_object_name, str(object_uri.generation))
    existing_components = [existing_component]
    enc_key_sha256 = TEST_ENCRYPTION_KEY1_SHA256_B64
    WriteParallelUploadTrackerFile(tracker_file_name, tracker_prefix, existing_components, encryption_key_sha256=enc_key_sha256)
    try:
        with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'parallel_composite_upload_component_size', str(component_size)), ('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]):
            stderr = self.RunGsUtil(['cp', source_file, suri(bucket_uri, 'foo')], return_stderr=True)
            self.assertIn('Found 1 existing temporary components to reuse.', stderr)
            self.assertFalse(os.path.exists(tracker_file_name), 'Tracker file %s should have been deleted.' % tracker_file_name)
            read_contents = self.RunGsUtil(['cat', suri(bucket_uri, 'foo')], return_stdout=True)
            self.assertEqual(read_contents.encode('ascii'), file_contents)
    finally:
        DeleteTrackerFile(tracker_file_name)