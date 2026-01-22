from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(subprocess, 'run', autospec=True)
def test_iam_ch_expands_urls_with_recursion_and_ignores_container_headers(self, mock_run):
    original_policy = {'bindings': [{'role': 'modified-role', 'members': ['allUsers']}]}
    with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
        ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/dir/', 'type': 'prefix'}, {'url': 'gs://b/dir/:', 'type': 'cloud_object'}, {'url': 'gs://b/dir2/', 'type': 'prefix'}, {'url': 'gs://b/dir2/o', 'type': 'cloud_object'}]))
        get_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(original_policy))
        set_process = subprocess.CompletedProcess(args=[], returncode=0)
        mock_run.side_effect = [ls_process] + [get_process, set_process] * 3
        with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            self.RunCommand('iam', ['ch', '-r', 'allAuthenticatedUsers:modified-role', 'gs://b'])
        self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', '-r', 'gs://b/']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/dir/:', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/dir/:', '-'], stdin=mock.ANY), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/dir2/o', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'set-iam-policy', 'gs://b/dir2/o', '-'], stdin=mock.ANY)])