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
def test_get_one_no_networks(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one('_test-cloud-domain-scoped_')
    self.assertEqual([], cc.get_external_networks())
    self.assertEqual([], cc.get_internal_networks())
    self.assertIsNone(cc.get_nat_source())
    self.assertIsNone(cc.get_nat_destination())
    self.assertIsNone(cc.get_default_network())