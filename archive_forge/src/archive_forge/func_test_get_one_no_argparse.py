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
def test_get_one_no_argparse(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one(cloud='_test-cloud_', argparse=None)
    self._assert_cloud_details(cc)
    self.assertEqual(cc.region_name, 'test-region')
    self.assertIsNone(cc.snack_type)