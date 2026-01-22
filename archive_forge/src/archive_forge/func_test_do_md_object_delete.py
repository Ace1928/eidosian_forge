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
def test_do_md_object_delete(self):
    args = self._make_args({'namespace': 'MyNamespace', 'object': 'MyObject'})
    with mock.patch.object(self.gc.metadefs_object, 'delete') as mocked_delete:
        test_shell.do_md_object_delete(self.gc, args)
        mocked_delete.assert_called_once_with('MyNamespace', 'MyObject')