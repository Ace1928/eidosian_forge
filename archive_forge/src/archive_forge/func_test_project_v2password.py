import argparse
import copy
import os
from unittest import mock
import fixtures
import testtools
import yaml
from openstack import config
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
def test_project_v2password(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'auth_type': 'v2password', 'auth': {'project-name': 'my_project_name', 'project-id': 'my_project_id'}}
    result = c._fix_backwards_project(cloud)
    expected = {'auth_type': 'v2password', 'auth': {'tenant_name': 'my_project_name', 'tenant_id': 'my_project_id'}}
    self.assertEqual(expected, result)