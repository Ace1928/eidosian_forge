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
def test_do_md_tag_create_multiple(self):
    args = self._make_args({'namespace': 'MyNamespace', 'delim': ',', 'names': 'MyTag1, MyTag2', 'append': False})
    with mock.patch.object(self.gc.metadefs_tag, 'create_multiple') as mocked_create_tags:
        expect_tags = [{'tags': [{'name': 'MyTag1'}, {'name': 'MyTag2'}]}]
        mocked_create_tags.return_value = expect_tags
        test_shell.do_md_tag_create_multiple(self.gc, args)
        mocked_create_tags.assert_called_once_with('MyNamespace', tags=['MyTag1', 'MyTag2'], append=False)
        utils.print_list.assert_called_once_with(expect_tags, ['name'], field_settings={'description': {'align': 'l', 'max_width': 50}})