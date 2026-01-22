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
def test_get_region_many_regions_default(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.no_yaml])
    region = c._get_region(cloud='_test_cloud_regions', region_name='')
    self.assertEqual(region, {'name': 'region1', 'values': {'external_network': 'region1-network'}})