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
@mock.patch('glanceclient.shell.OpenStackImagesShell._get_keystone_auth_plugin')
@mock.patch.object(openstack_shell.OpenStackImagesShell, '_cache_schemas', return_value=False)
def test_no_auth_with_proj_name(self, cache_schemas, session):
    with mock.patch('glanceclient.v2.client.Client'):
        args = '--os-project-name myname --os-project-domain-name mydomain --os-project-domain-id myid --os-image-api-version 2 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        args, kwargs = session.call_args
        self.assertEqual('myname', kwargs['project_name'])
        self.assertEqual('mydomain', kwargs['project_domain_name'])
        self.assertEqual('myid', kwargs['project_domain_id'])