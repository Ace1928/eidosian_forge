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
def test_do_md_namespace_resource_type_list(self):
    args = self._make_args({'namespace': 'MyNamespace'})
    with mock.patch.object(self.gc.metadefs_resource_type, 'get') as mocked_get:
        expect_objects = [{'namespace': 'MyNamespace', 'object': 'MyObject'}]
        mocked_get.return_value = expect_objects
        test_shell.do_md_namespace_resource_type_list(self.gc, args)
        mocked_get.assert_called_once_with('MyNamespace')
        utils.print_list.assert_called_once_with(expect_objects, ['name', 'prefix', 'properties_target'])