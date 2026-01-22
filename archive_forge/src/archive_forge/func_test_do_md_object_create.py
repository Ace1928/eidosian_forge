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
def test_do_md_object_create(self):
    args = self._make_args({'namespace': 'MyNamespace', 'name': 'MyObject', 'schema': '{}'})
    with mock.patch.object(self.gc.metadefs_object, 'create') as mocked_create:
        expect_object = {'namespace': 'MyNamespace', 'name': 'MyObject'}
        mocked_create.return_value = expect_object
        test_shell.do_md_object_create(self.gc, args)
        mocked_create.assert_called_once_with('MyNamespace', name='MyObject')
        utils.print_dict.assert_called_once_with(expect_object)