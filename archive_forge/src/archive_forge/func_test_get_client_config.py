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
def test_get_client_config(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one(cloud='_test_cloud_regions')
    defaults = {'use_hostnames': False, 'other-value': 'something', 'force_ipv4': False}
    ansible_options = cc.get_client_config('ansible', defaults)
    self.assertDictEqual({'expand_hostvars': False, 'use_hostnames': True, 'other_value': 'something', 'force_ipv4': True}, ansible_options)