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
def test_metrics_global(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.secure_yaml])
    self.assertIsInstance(c.cloud_config, dict)
    cc = c.get_one('_test-cloud_')
    statsd = {'host': '127.0.0.1', 'port': '1234'}
    self.assertEqual(statsd['host'], cc._statsd_host)
    self.assertEqual(statsd['port'], cc._statsd_port)
    self.assertEqual('openstack.api', cc.get_statsd_prefix())
    influxdb = {'use_udp': True, 'host': '127.0.0.1', 'port': '1234', 'username': 'username', 'password': 'password', 'database': 'database', 'measurement': 'measurement.name', 'timeout': 10}
    self.assertEqual(influxdb, cc._influxdb_config)