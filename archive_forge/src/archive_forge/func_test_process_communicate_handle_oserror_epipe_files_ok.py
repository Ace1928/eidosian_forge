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
@mock.patch('keystoneclient.common.cms._check_files_accessible')
def test_process_communicate_handle_oserror_epipe_files_ok(self, files_acc_mock):
    process_mock = mock.Mock()
    process_mock.communicate = self._raise_OSError
    process_mock.stderr = mock.Mock()
    process_mock.stderr.read = mock.Mock(return_value='proc stderr')
    files_acc_mock.return_value = (-1, None)
    output, err, retcode = cms._process_communicate_handle_oserror(process_mock, '', [])
    self.assertEqual((output, retcode), ('', -1))
    self.assertIn('proc stderr', err)