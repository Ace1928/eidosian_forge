from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from gslib.commands import acl
from gslib.command import CreateOrGetGsutilLogger
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils import acl_helper
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_acl_set_version_specific_uri(self):
    """Tests setting an ACL on a specific version of an object."""
    bucket_uri = self.CreateVersionedBucket()
    uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data')
    inpath = self.CreateTempFile(contents=b'def')
    self.RunGsUtil(['cp', inpath, uri.uri])
    lines = self.AssertNObjectsInBucket(bucket_uri, 2, versioned=True)
    v0_uri_str, v1_uri_str = (lines[0], lines[1])
    orig_acls = []
    for uri_str in (v0_uri_str, v1_uri_str):
        acl = self.RunGsUtil(self._get_acl_prefix + [uri_str], return_stdout=True)
        self.assertNotIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
        orig_acls.append(acl)
    self.RunGsUtil(self._set_acl_prefix + ['public-read', v0_uri_str])
    acl = self.RunGsUtil(self._get_acl_prefix + [v0_uri_str], return_stdout=True)
    self.assertIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
    acl = self.RunGsUtil(self._get_acl_prefix + [v1_uri_str], return_stdout=True)
    self.assertNotIn(PUBLIC_READ_JSON_ACL_TEXT, self._strip_json_whitespace(acl))
    acl = self.RunGsUtil(self._get_acl_prefix + [uri.uri], return_stdout=True)
    self.assertEqual(acl, orig_acls[0])