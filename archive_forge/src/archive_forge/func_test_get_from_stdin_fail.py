import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
@mock.patch.object(sys, 'stdin', autospec=True)
def test_get_from_stdin_fail(self, mock_stdin):
    mock_stdin.read.side_effect = IOError
    desc = 'something'
    self.assertRaises(exc.InvalidAttribute, utils.get_from_stdin, desc)
    mock_stdin.read.assert_called_once_with()