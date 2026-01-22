import argparse
import copy
import os
import extras
import fixtures
import testtools
import yaml
from openstack.config import defaults
from os_client_config import cloud_config
from os_client_config import config
from os_client_config import exceptions
from os_client_config.tests import base
def test_get_one_cloud_no_yaml_no_cloud(self):
    c = config.OpenStackConfig(load_yaml_config=False)
    self.assertRaises(exceptions.OpenStackConfigException, c.get_one_cloud, cloud='_test_cloud_regions', region_name='region2', argparse=None)