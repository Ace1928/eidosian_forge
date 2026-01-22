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
def test_process_communicate_handle_oserror_no_exception(self):
    process_mock = mock.Mock()
    process_mock.communicate.return_value = ('out', 'err')
    process_mock.poll.return_value = 0
    output, err, retcode = cms._process_communicate_handle_oserror(process_mock, '', [])
    self.assertEqual(output, 'out')
    self.assertEqual(err, 'err')
    self.assertEqual(retcode, 0)