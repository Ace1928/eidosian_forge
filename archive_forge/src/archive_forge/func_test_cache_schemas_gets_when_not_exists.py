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
@mock.patch('os.path.exists', side_effect=[True, False, False, False])
def test_cache_schemas_gets_when_not_exists(self, exists_mock):
    options = {'get_schema': False, 'os_auth_url': self.os_auth_url}
    schema_odict = OrderedDict(self.schema_dict)
    args = self._make_args(options)
    client = self.shell._get_versioned_client('2', args)
    self.shell._cache_schemas(args, client, home_dir=self.cache_dir)
    self.assertEqual(12, open.mock_calls.__len__())
    self.assertEqual(mock.call(self.cache_files[0], 'w'), open.mock_calls[0])
    self.assertEqual(mock.call(self.cache_files[1], 'w'), open.mock_calls[4])
    actual = json.loads(open.mock_calls[2][1][0])
    self.assertEqual(schema_odict, actual)
    actual = json.loads(open.mock_calls[6][1][0])
    self.assertEqual(schema_odict, actual)