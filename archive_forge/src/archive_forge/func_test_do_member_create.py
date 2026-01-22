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
def test_do_member_create(self):
    args = self._make_args({'image_id': 'IMG-01', 'member_id': 'MEM-01'})
    with mock.patch.object(self.gc.image_members, 'create') as mock_create:
        mock_create.return_value = {}
        test_shell.do_member_create(self.gc, args)
        mock_create.assert_called_once_with('IMG-01', 'MEM-01')
        columns = ['Image ID', 'Member ID', 'Status']
        utils.print_list.assert_called_once_with([{}], columns)