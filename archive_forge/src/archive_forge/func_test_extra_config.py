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
def test_extra_config(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    defaults = {'use_hostnames': False, 'other-value': 'something'}
    ansible_options = c.get_extra_config('ansible', defaults)
    self.assertDictEqual({'expand_hostvars': False, 'use_hostnames': True, 'other_value': 'something'}, ansible_options)