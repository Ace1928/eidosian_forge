import io
import sys
from unittest import mock
from keystoneauth1 import exceptions
import testtools
from aodhclient import shell
@mock.patch('sys.stderr', io.StringIO())
def test_cli_http_error_with_details(self):
    shell.AodhShell().clean_up(None, None, exceptions.HttpError('foo', details='bar'))
    stderr_lines = sys.stderr.getvalue().splitlines()
    self.assertEqual(1, len(stderr_lines))
    self.assertEqual('bar', stderr_lines[0])