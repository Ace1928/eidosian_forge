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
def test_cms_verify_token_no_oserror(self):
    with mock.patch('subprocess.Popen.communicate', new=self._raise_OSError):
        try:
            cms.cms_verify('x', '/no/such/file', '/no/such/key')
        except exceptions.CertificateConfigError as e:
            self.assertIn('/no/such/file', e.output)
            self.assertIn('Hit OSError ', e.output)
        else:
            self.fail('Expected exceptions.CertificateConfigError')