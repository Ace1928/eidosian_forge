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
def test_cms_verify_token_no_files(self):
    self.assertRaises(exceptions.CertificateConfigError, cms.cms_verify, self.examples.SIGNED_TOKEN_SCOPED, '/no/such/file', '/no/such/key')