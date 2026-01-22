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
def test_get_all_clouds(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.no_yaml])
    clouds = c.get_all_clouds()
    user_clouds = [cloud for cloud in base.USER_CONF['clouds'].keys()] + ['_test_cloud_regions', '_test_cloud_regions']
    configured_clouds = [cloud.name for cloud in clouds]
    self.assertCountEqual(user_clouds, configured_clouds)