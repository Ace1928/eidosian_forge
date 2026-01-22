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
def test_do_md_tag_update(self):
    args = self._make_args({'namespace': 'MyNamespace', 'tag': 'MyTag', 'name': 'NewTag'})
    with mock.patch.object(self.gc.metadefs_tag, 'update') as mocked_update:
        expect_tag = {'namespace': 'MyNamespace', 'name': 'NewTag'}
        mocked_update.return_value = expect_tag
        test_shell.do_md_tag_update(self.gc, args)
        mocked_update.assert_called_once_with('MyNamespace', 'MyTag', name='NewTag')
        utils.print_dict.assert_called_once_with(expect_tag)