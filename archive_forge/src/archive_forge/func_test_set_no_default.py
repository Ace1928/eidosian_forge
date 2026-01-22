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
def test_set_no_default(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cloud = {'identity_endpoint_type': 'admin', 'compute_endpoint_type': 'private', 'endpoint_type': 'public', 'auth_type': 'v3password'}
    result = c._fix_backwards_interface(cloud)
    expected = {'identity_interface': 'admin', 'compute_interface': 'private', 'interface': 'public', 'auth_type': 'v3password'}
    self.assertDictEqual(expected, result)