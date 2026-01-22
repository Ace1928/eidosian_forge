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
def test_get_one_no_yaml_no_cloud(self):
    c = config.OpenStackConfig(load_yaml_config=False)
    self.assertRaises(exceptions.ConfigException, c.get_one, cloud='_test_cloud_regions', region_name='region2', argparse=None)