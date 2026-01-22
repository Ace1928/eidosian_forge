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
def test_cms_verify(self):
    self.assertRaises(exceptions.CertificateConfigError, cms.cms_verify, 'data', 'no_exist_cert_file', 'no_exist_ca_file')