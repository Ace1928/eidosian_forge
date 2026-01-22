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
def test_subcommand_help(self):
    stdout, stderr = self.shell('help stores-delete')
    expected = 'usage: glance stores-delete --store <STORE_ID> <IMAGE_ID>\n\nDelete image from specific store.\n\nPositional arguments:\n  <IMAGE_ID>          ID of image to update.\n\nRequired arguments:\n  --store <STORE_ID>  Store to delete image from.\n'
    self.assertEqual(expected, stdout)