import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
@mock.patch('glanceclient.common.utils.exit')
@mock.patch('sys.stdin', autospec=True)
def test_neg_stores_info(self, mock_stdin, mock_utils_exit):
    expected_msg = 'Multi Backend support is not enabled'
    args = self._make_args({'detail': False})
    mock_utils_exit.side_effect = self._mock_utils_exit
    with mock.patch.object(self.gc.images, 'get_stores_info') as mocked_info:
        mocked_info.side_effect = exc.HTTPNotFound
        try:
            test_shell.do_stores_info(self.gc, args)
            self.fail('utils.exit should have been called')
        except SystemExit:
            pass
    mock_utils_exit.assert_called_once_with(expected_msg)