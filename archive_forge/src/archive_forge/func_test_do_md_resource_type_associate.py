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
def test_do_md_resource_type_associate(self):
    args = self._make_args({'namespace': 'MyNamespace', 'name': 'MyResourceType', 'prefix': 'PREFIX:'})
    with mock.patch.object(self.gc.metadefs_resource_type, 'associate') as mocked_associate:
        expect_rt = {'namespace': 'MyNamespace', 'name': 'MyResourceType', 'prefix': 'PREFIX:'}
        mocked_associate.return_value = expect_rt
        test_shell.do_md_resource_type_associate(self.gc, args)
        mocked_associate.assert_called_once_with('MyNamespace', **expect_rt)
        utils.print_dict.assert_called_once_with(expect_rt)