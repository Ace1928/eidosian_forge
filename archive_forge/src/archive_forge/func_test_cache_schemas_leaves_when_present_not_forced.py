import argparse
from collections import OrderedDict
import hashlib
import io
import logging
import os
import sys
import traceback
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import fixture as ks_fixture
from requests_mock.contrib import fixture as rm_fixture
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell as openstack_shell
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import image_versions_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2 import schemas as schemas
import json
@mock.patch('builtins.open', new=mock.mock_open(), create=True)
@mock.patch('os.path.exists', return_value=True)
def test_cache_schemas_leaves_when_present_not_forced(self, exists_mock):
    options = {'get_schema': False, 'os_auth_url': self.os_auth_url}
    client = mock.MagicMock()
    self.shell._cache_schemas(self._make_args(options), client, home_dir=self.cache_dir)
    exists_mock.assert_has_calls([mock.call(self.prefix_path), mock.call(self.cache_files[0]), mock.call(self.cache_files[1]), mock.call(self.cache_files[2])])
    self.assertEqual(4, exists_mock.call_count)
    self.assertEqual(0, open.mock_calls.__len__())