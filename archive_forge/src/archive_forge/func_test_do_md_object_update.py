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
def test_do_md_object_update(self):
    args = self._make_args({'namespace': 'MyNamespace', 'object': 'MyObject', 'name': 'NewName', 'schema': '{}'})
    with mock.patch.object(self.gc.metadefs_object, 'update') as mocked_update:
        expect_object = {'namespace': 'MyNamespace', 'name': 'MyObject'}
        mocked_update.return_value = expect_object
        test_shell.do_md_object_update(self.gc, args)
        mocked_update.assert_called_once_with('MyNamespace', 'MyObject', name='NewName')
        utils.print_dict.assert_called_once_with(expect_object)