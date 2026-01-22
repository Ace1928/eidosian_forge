from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import stat
from gslib.exception import CommandException
from gslib.parallel_tracker_file import ObjectFromTracker
from gslib.parallel_tracker_file import ReadParallelUploadTrackerFile
from gslib.parallel_tracker_file import ValidateParallelCompositeTrackerData
from gslib.parallel_tracker_file import WriteComponentToParallelUploadTrackerFile
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.storage_url import StorageUrlFromString
from gslib.tests.testcase.unit_testcase import GsUtilUnitTestCase
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import _HashFilename
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import HashRewriteParameters
from gslib.tracker_file import ReadRewriteTrackerFile
from gslib.tracker_file import WriteRewriteTrackerFile
from gslib.utils import parallelism_framework_util
from gslib.utils.constants import UTF8
def testValidateParallelCompositeTrackerData(self):
    tempdir = self.CreateTempDir()
    fpath = os.path.join(tempdir, 'foo')
    random_prefix = '123'
    old_enc_key = '456'
    bucket_url = StorageUrlFromString('gs://foo')
    objects = [ObjectFromTracker('obj1', '42'), ObjectFromTracker('obj2', '314159')]
    WriteParallelUploadTrackerFile(fpath, random_prefix, objects, encryption_key_sha256=old_enc_key)
    if os.name == 'posix':
        mode = oct(stat.S_IMODE(os.stat(fpath).st_mode))
        self.assertEqual(oct(384), mode)

    class MockCommandObject(object):
        delete_called = False

        class ParallelOverrideReason(object):
            SPEED = 'speed'

        def Apply(self, *unused_args, **unused_kwargs):
            self.delete_called = True

    def MockDeleteFunc():
        pass

    def MockDeleteExceptionHandler():
        pass
    command_obj = MockCommandObject()
    actual_prefix, actual_objects = ValidateParallelCompositeTrackerData(fpath, old_enc_key, random_prefix, objects, old_enc_key, bucket_url, command_obj, self.logger, MockDeleteFunc, MockDeleteExceptionHandler)
    self.assertEqual(False, command_obj.delete_called)
    self.assertEqual(random_prefix, actual_prefix)
    self.assertEqual(objects, actual_objects)
    new_enc_key = '789'
    command_obj = MockCommandObject()
    actual_prefix, actual_objects = ValidateParallelCompositeTrackerData(fpath, old_enc_key, random_prefix, objects, new_enc_key, bucket_url, command_obj, self.logger, MockDeleteFunc, MockDeleteExceptionHandler)
    self.assertEqual(True, command_obj.delete_called)
    self.assertEqual(None, actual_prefix)
    self.assertEqual([], actual_objects)