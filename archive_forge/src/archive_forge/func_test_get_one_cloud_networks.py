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
def test_get_one_cloud_networks(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one_cloud('_test-cloud-networks_')
    self.assertEqual(['a-public', 'another-public', 'split-default'], cc.get_external_networks())
    self.assertEqual(['a-private', 'another-private', 'split-no-default'], cc.get_internal_networks())
    self.assertEqual('a-public', cc.get_nat_source())
    self.assertEqual('another-private', cc.get_nat_destination())
    self.assertEqual('another-public', cc.get_default_network())
    self.assertEqual(['a-public', 'another-public', 'split-no-default'], cc.get_external_ipv4_networks())
    self.assertEqual(['a-public', 'another-public', 'split-default'], cc.get_external_ipv6_networks())