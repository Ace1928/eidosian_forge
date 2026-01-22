import errno
import os
import subprocess
from unittest import mock
import testresources
from testtools import matchers
from keystoneclient.common import cms
from keystoneclient import exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
def test_cms_sign_token_no_files_pkiz(self):
    self.assertRaises(subprocess.CalledProcessError, cms.pkiz_sign, self.examples.TOKEN_SCOPED_DATA, '/no/such/file', '/no/such/key')